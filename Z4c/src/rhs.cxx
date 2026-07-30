#include <AMReX_GpuAsyncArray.H>

#include <cctk.h>

#ifdef __CUDACC__
// Disable CCTK_DEBUG since the debug information takes too much
// parameter space to launch the kernels
#ifdef CCTK_DEBUG
#undef CCTK_DEBUG
#endif
#endif

#include "derivs.hxx"
#include "physics.hxx"
#include "z4c_vars.hxx"

#include <loop_device.hxx>
#include <mat.hxx>
#include <simd.hxx>
#include <vec.hxx>

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#ifdef __CUDACC__
#include <nvtx3/nvToolsExt.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>

namespace Z4c {
using namespace Arith;
using namespace Loop;
using namespace std;

constexpr int max_eta_sources = 100;

struct PunctureEtaSource {
  CCTK_REAL x;
  CCTK_REAL y;
  CCTK_REAL z;
  CCTK_REAL mass;
  CCTK_REAL weight;
};

struct PunctureEtaProfile {
  bool enabled = false;
  bool radial_baseline = false;
  CCTK_INT num_sources = 0;
  const PunctureEtaSource *sources = nullptr;
  CCTK_REAL eta0 = 0.0;
  CCTK_REAL reference_mass = 1.0;
  CCTK_REAL coefficient = 1.0;
  CCTK_INT power = 1;
  CCTK_REAL eta_min = 0.0;
  CCTK_REAL eta_max = 0.0;
};

ARITH_DEVICE ARITH_INLINE CCTK_REAL clamp_eta(const CCTK_REAL eta_value,
                                              const CCTK_REAL eta_min,
                                              const CCTK_REAL eta_max) {
  return eta_value < eta_min ? eta_min
                             : (eta_value > eta_max ? eta_max : eta_value);
}

ARITH_DEVICE ARITH_INLINE CCTK_REAL radial_eta(
    const CCTK_REAL x, const CCTK_REAL y, const CCTK_REAL z,
    const CCTK_REAL veta_width, const CCTK_REAL veta_central,
    const CCTK_REAL veta_outer) {
  const CCTK_REAL r2 = x * x + y * y + z * z;
  const CCTK_REAL r4 = r2 * r2;
  const CCTK_REAL w4 =
      veta_width * veta_width * veta_width * veta_width;
  return (veta_central - veta_outer) * exp(-r4 / w4) + veta_outer;
}

ARITH_DEVICE ARITH_INLINE CCTK_REAL
integer_power(const CCTK_REAL base, const CCTK_INT power) {
  CCTK_REAL result = 1.0;
  for (int n = 0; n < power; ++n)
    result *= base;
  return result;
}

ARITH_DEVICE ARITH_INLINE CCTK_REAL
puncture_tracker_eta(const PunctureEtaProfile &profile, const CCTK_REAL x,
                     const CCTK_REAL y, const CCTK_REAL z,
                     const CCTK_REAL fallback_eta) {
  if (!profile.enabled)
    return fallback_eta;

  const CCTK_REAL baseline =
      profile.radial_baseline ? fallback_eta : profile.eta0;

  CCTK_REAL eta_value = baseline;
  const CCTK_REAL inv_m0_sq =
      1.0 / (profile.reference_mass * profile.reference_mass);
  for (int n = 0; n < profile.num_sources; ++n) {
    const PunctureEtaSource &source = profile.sources[n];

    const CCTK_REAL dx = x - source.x;
    const CCTK_REAL dy = y - source.y;
    const CCTK_REAL dz = z - source.z;
    const CCTK_REAL r2 = dx * dx + dy * dy + dz * dz;
    const CCTK_REAL rhat2 = r2 * inv_m0_sq;
    const CCTK_REAL denominator =
        1.0 + source.weight * integer_power(rhat2, profile.power);

    eta_value += profile.coefficient *
                 (1.0 / source.mass - baseline) / denominator;
  }

  return clamp_eta(eta_value, profile.eta_min, profile.eta_max);
}

ARITH_DEVICE ARITH_INLINE CCTK_REAL eta_at_point(
    const PunctureEtaProfile &profile, const CCTK_REAL x, const CCTK_REAL y,
    const CCTK_REAL z, const CCTK_REAL veta_width,
    const CCTK_REAL veta_central, const CCTK_REAL veta_outer) {
  if (!profile.enabled)
    return radial_eta(x, y, z, veta_width, veta_central, veta_outer);

  const CCTK_REAL fallback_eta =
      profile.radial_baseline
          ? radial_eta(x, y, z, veta_width, veta_central, veta_outer)
          : profile.eta0;

  return puncture_tracker_eta(profile, x, y, z, fallback_eta);
}

extern "C" void Z4c_RHS(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS_Z4c_RHS;
  DECLARE_CCTK_PARAMETERS;

  PunctureEtaProfile eta_profile_data;
  array<PunctureEtaSource, max_eta_sources> eta_sources{};
  eta_profile_data.enabled =
      CCTK_EQUALS(eta_profile, "puncture_tracker") ||
      CCTK_EQUALS(eta_profile, "radial_puncture_tracker");
  eta_profile_data.radial_baseline =
      CCTK_EQUALS(eta_profile, "radial_puncture_tracker");
  eta_profile_data.num_sources = 0;
  eta_profile_data.eta0 = eta;
  eta_profile_data.reference_mass = eta_reference_mass;
  eta_profile_data.coefficient = eta_profile_coefficient;
  eta_profile_data.power = eta_profile_power;
  eta_profile_data.eta_min = eta_profile_min;
  eta_profile_data.eta_max = eta_profile_max;

  if (eta_profile_data.eta_min > eta_profile_data.eta_max)
    CCTK_VERROR("eta_profile_min=%g is larger than eta_profile_max=%g",
                double(eta_profile_data.eta_min),
                double(eta_profile_data.eta_max));

  if (eta_profile_data.enabled) {
    const auto pt_num_tracked_ptr = static_cast<const CCTK_INT *>(
        CCTK_VarDataPtr(cctkGH, 0, "PunctureTracker::pt_num_tracked[0]"));
    const auto pt_num_groups_ptr = static_cast<const CCTK_INT *>(
        CCTK_VarDataPtr(cctkGH, 0, "PunctureTracker::pt_num_groups[0]"));
    const auto pt_mass_ptr = static_cast<const CCTK_REAL *>(
        CCTK_VarDataPtr(cctkGH, 0, "PunctureTracker::pt_mass[0]"));
    const auto pt_group_x_ptr = static_cast<const CCTK_REAL *>(
        CCTK_VarDataPtr(cctkGH, 0, "PunctureTracker::pt_group_x[0]"));
    const auto pt_group_y_ptr = static_cast<const CCTK_REAL *>(
        CCTK_VarDataPtr(cctkGH, 0, "PunctureTracker::pt_group_y[0]"));
    const auto pt_group_z_ptr = static_cast<const CCTK_REAL *>(
        CCTK_VarDataPtr(cctkGH, 0, "PunctureTracker::pt_group_z[0]"));
    const auto pt_group_mass_ptr = static_cast<const CCTK_REAL *>(
        CCTK_VarDataPtr(cctkGH, 0, "PunctureTracker::pt_group_mass[0]"));
    const auto pt_group_eta_weight_ptr = static_cast<const CCTK_REAL *>(
        CCTK_VarDataPtr(cctkGH, 0,
                        "PunctureTracker::pt_group_eta_weight[0]"));

    if (!pt_num_tracked_ptr || !pt_num_groups_ptr || !pt_mass_ptr ||
        !pt_group_x_ptr || !pt_group_y_ptr || !pt_group_z_ptr ||
        !pt_group_mass_ptr || !pt_group_eta_weight_ptr)
      CCTK_ERROR("PunctureTracker eta profiles require active "
                 "PunctureTracker grouped source scalars");

    if (!isfinite(eta_profile_data.coefficient))
      CCTK_VERROR("eta_profile_coefficient=%g must be finite",
                  double(eta_profile_data.coefficient));

    const CCTK_INT num_tracked = pt_num_tracked_ptr[0];
    const CCTK_INT num_groups = pt_num_groups_ptr[0];
    if (num_tracked <= 0 || num_tracked > max_eta_sources)
      CCTK_VERROR("PunctureTracker::pt_num_tracked=%d is invalid",
                  int(num_tracked));
    if (num_groups <= 0 || num_groups > max_eta_sources)
      CCTK_VERROR("PunctureTracker::pt_num_groups=%d is invalid",
                  int(num_groups));

    CCTK_REAL minimum_mass = pt_mass_ptr[0];
    for (int n = 0; n < num_tracked; ++n) {
      if (!isfinite(pt_mass_ptr[n]) || pt_mass_ptr[n] <= 0.0)
        CCTK_VERROR("PunctureTracker::pt_mass[%d]=%g must be positive and "
                    "finite",
                    n, double(pt_mass_ptr[n]));
      minimum_mass =
          n == 0 ? pt_mass_ptr[n] : std::min(minimum_mass, pt_mass_ptr[n]);
    }

    eta_profile_data.reference_mass =
        eta_profile_data.reference_mass > 0.0
            ? eta_profile_data.reference_mass
            : minimum_mass;
    if (!isfinite(eta_profile_data.reference_mass) ||
        eta_profile_data.reference_mass <= 0.0)
      CCTK_VERROR("eta_reference_mass=%g must be positive when specified",
                  double(eta_profile_data.reference_mass));

    eta_profile_data.num_sources = num_groups;
    for (int n = 0; n < eta_profile_data.num_sources; ++n) {
      PunctureEtaSource &source = eta_sources[n];
      source.x = pt_group_x_ptr[n];
      source.y = pt_group_y_ptr[n];
      source.z = pt_group_z_ptr[n];
      source.mass = pt_group_mass_ptr[n];
      source.weight = pt_group_eta_weight_ptr[n];

      if (!isfinite(source.x) || !isfinite(source.y) ||
          !isfinite(source.z))
        CCTK_VERROR("PunctureTracker grouped source %d is not finite: "
                    "(%g,%g,%g)",
                    n, double(source.x), double(source.y), double(source.z));

      if (!isfinite(source.mass) || source.mass <= 0.0)
        CCTK_VERROR("PunctureTracker::pt_group_mass[%d]=%g must be positive "
                    "and finite",
                    n, double(source.mass));

      if (!isfinite(source.weight) || source.weight < 0.0)
        CCTK_VERROR("PunctureTracker::pt_group_eta_weight[%d]=%g must be "
                    "non-negative and finite",
                    n, double(source.weight));
    }
  }

  amrex::Gpu::AsyncArray<PunctureEtaSource> eta_source_storage(
      eta_sources.data(), eta_profile_data.num_sources);
  eta_profile_data.sources = eta_source_storage.data();

  for (int d = 0; d < 3; ++d)
    if (cctk_nghostzones[d] < deriv_order / 2 + 1)
      CCTK_VERROR("Need at least %d ghost zones", deriv_order / 2 + 1);

  //

  const array<int, dim> indextype = {0, 0, 0};
  const array<int, dim> nghostzones = {cctk_nghostzones[0], cctk_nghostzones[1],
                                       cctk_nghostzones[2]};
  vect<int, dim> imin, imax;
  GridDescBase(cctkGH).box_int<0, 0, 0>(nghostzones, imin, imax);
  // Suffix 1: with ghost zones, suffix 0: without ghost zones
  const GF3D2layout layout1(cctkGH, indextype);
  const GF3D5layout layout0(imin, imax);

  const GF3D2<const CCTK_REAL> gf_chi1(layout1, chi);

  const smat<GF3D2<const CCTK_REAL>, 3> gf_gammat1{
      GF3D2<const CCTK_REAL>(layout1, gammatxx),
      GF3D2<const CCTK_REAL>(layout1, gammatxy),
      GF3D2<const CCTK_REAL>(layout1, gammatxz),
      GF3D2<const CCTK_REAL>(layout1, gammatyy),
      GF3D2<const CCTK_REAL>(layout1, gammatyz),
      GF3D2<const CCTK_REAL>(layout1, gammatzz)};

  const GF3D2<const CCTK_REAL> gf_Kh1(layout1, Kh);

  const smat<GF3D2<const CCTK_REAL>, 3> gf_At1{
      GF3D2<const CCTK_REAL>(layout1, Atxx),
      GF3D2<const CCTK_REAL>(layout1, Atxy),
      GF3D2<const CCTK_REAL>(layout1, Atxz),
      GF3D2<const CCTK_REAL>(layout1, Atyy),
      GF3D2<const CCTK_REAL>(layout1, Atyz),
      GF3D2<const CCTK_REAL>(layout1, Atzz)};

  const vec<GF3D2<const CCTK_REAL>, 3> gf_Gamt1{
      GF3D2<const CCTK_REAL>(layout1, Gamtx),
      GF3D2<const CCTK_REAL>(layout1, Gamty),
      GF3D2<const CCTK_REAL>(layout1, Gamtz)};

  const GF3D2<const CCTK_REAL> gf_Theta1(layout1, Theta);

  const GF3D2<const CCTK_REAL> gf_alphaG1(layout1, alphaG);

  const vec<GF3D2<const CCTK_REAL>, 3> gf_betaG1{
      GF3D2<const CCTK_REAL>(layout1, betaGx),
      GF3D2<const CCTK_REAL>(layout1, betaGy),
      GF3D2<const CCTK_REAL>(layout1, betaGz)};

  //

  // Ideas:
  //
  // - Outline certain functions, e.g. `det` or `raise_index`. Ensure
  //   they are called with floating-point arguments, not tensor
  //   indices.

  const int ntmps = 154;
  GF3D5vector<CCTK_REAL> tmps(layout0, ntmps);
  int itmp = 0;

  const auto make_gf = [&]() { return GF3D5<CCTK_REAL>(tmps(itmp++)); };
  const auto make_vec = [&](const auto &f) {
    return vec<result_of_t<decltype(f)()>, 3>([&](int) { return f(); });
  };
  const auto make_mat = [&](const auto &f) {
    return smat<result_of_t<decltype(f)()>, 3>([&](int, int) { return f(); });
  };
  const auto make_vec_gf = [&]() { return make_vec(make_gf); };
  const auto make_mat_gf = [&]() { return make_mat(make_gf); };
  const auto make_vec_vec_gf = [&]() { return make_vec(make_vec_gf); };
  const auto make_vec_mat_gf = [&]() { return make_vec(make_mat_gf); };
  const auto make_mat_vec_gf = [&]() { return make_mat(make_vec_gf); };
  const auto make_mat_mat_gf = [&]() { return make_mat(make_mat_gf); };

  const GF3D5<CCTK_REAL> gf_chi0(make_gf());
  const vec<GF3D5<CCTK_REAL>, 3> gf_dchi0(make_vec_gf());
  const smat<GF3D5<CCTK_REAL>, 3> gf_ddchi0(make_mat_gf());
  calc_derivs2(cctkGH, gf_chi1, gf_chi0, gf_dchi0, gf_ddchi0, layout0);

  const smat<GF3D5<CCTK_REAL>, 3> gf_gammat0(make_mat_gf());
  const smat<vec<GF3D5<CCTK_REAL>, 3>, 3> gf_dgammat0(make_mat_vec_gf());
  const smat<smat<GF3D5<CCTK_REAL>, 3>, 3> gf_ddgammat0(make_mat_mat_gf());
  calc_derivs2(cctkGH, gf_gammat1, gf_gammat0, gf_dgammat0, gf_ddgammat0,
               layout0);

  const GF3D5<CCTK_REAL> gf_Kh0(make_gf());
  const vec<GF3D5<CCTK_REAL>, 3> gf_dKh0(make_vec_gf());
  calc_derivs(cctkGH, gf_Kh1, gf_Kh0, gf_dKh0, layout0);

  const smat<GF3D5<CCTK_REAL>, 3> gf_At0(make_mat_gf());
  const smat<vec<GF3D5<CCTK_REAL>, 3>, 3> gf_dAt0(make_mat_vec_gf());
  calc_derivs(cctkGH, gf_At1, gf_At0, gf_dAt0, layout0);

  const vec<GF3D5<CCTK_REAL>, 3> gf_Gamt0(make_vec_gf());
  const vec<vec<GF3D5<CCTK_REAL>, 3>, 3> gf_dGamt0(make_vec_vec_gf());
  calc_derivs(cctkGH, gf_Gamt1, gf_Gamt0, gf_dGamt0, layout0);

  const GF3D5<CCTK_REAL> gf_Theta0(make_gf());
  const vec<GF3D5<CCTK_REAL>, 3> gf_dTheta0(make_vec_gf());
  calc_derivs(cctkGH, gf_Theta1, gf_Theta0, gf_dTheta0, layout0);

  const GF3D5<CCTK_REAL> gf_alphaG0(make_gf());
  const vec<GF3D5<CCTK_REAL>, 3> gf_dalphaG0(make_vec_gf());
  const smat<GF3D5<CCTK_REAL>, 3> gf_ddalphaG0(make_mat_gf());
  calc_derivs2(cctkGH, gf_alphaG1, gf_alphaG0, gf_dalphaG0, gf_ddalphaG0,
               layout0);

  const vec<GF3D5<CCTK_REAL>, 3> gf_betaG0(make_vec_gf());
  const vec<vec<GF3D5<CCTK_REAL>, 3>, 3> gf_dbetaG0(make_vec_vec_gf());
  const vec<smat<GF3D5<CCTK_REAL>, 3>, 3> gf_ddbetaG0(make_vec_mat_gf());
  calc_derivs2(cctkGH, gf_betaG1, gf_betaG0, gf_dbetaG0, gf_ddbetaG0, layout0);

  if (itmp != ntmps)
    CCTK_VERROR("Wrong number of temporary variables: ntmps=%d itmp=%d", ntmps,
                itmp);
  itmp = -1;

  //

  const GF3D2<const CCTK_REAL> gf_eTtt1(layout1, eTtt);

  const vec<GF3D2<const CCTK_REAL>, 3> gf_eTti1{
      GF3D2<const CCTK_REAL>(layout1, eTtx),
      GF3D2<const CCTK_REAL>(layout1, eTty),
      GF3D2<const CCTK_REAL>(layout1, eTtz)};

  const smat<GF3D2<const CCTK_REAL>, 3> gf_eTij1{
      GF3D2<const CCTK_REAL>(layout1, eTxx),
      GF3D2<const CCTK_REAL>(layout1, eTxy),
      GF3D2<const CCTK_REAL>(layout1, eTxz),
      GF3D2<const CCTK_REAL>(layout1, eTyy),
      GF3D2<const CCTK_REAL>(layout1, eTyz),
      GF3D2<const CCTK_REAL>(layout1, eTzz)};

  //

  const GF3D2<CCTK_REAL> gf_chi_rhs1(layout1, chi_rhs);

  const smat<GF3D2<CCTK_REAL>, 3> gf_gammat_rhs1{
      GF3D2<CCTK_REAL>(layout1, gammatxx_rhs),
      GF3D2<CCTK_REAL>(layout1, gammatxy_rhs),
      GF3D2<CCTK_REAL>(layout1, gammatxz_rhs),
      GF3D2<CCTK_REAL>(layout1, gammatyy_rhs),
      GF3D2<CCTK_REAL>(layout1, gammatyz_rhs),
      GF3D2<CCTK_REAL>(layout1, gammatzz_rhs)};

  const GF3D2<CCTK_REAL> gf_Kh_rhs1(layout1, Kh_rhs);

  const smat<GF3D2<CCTK_REAL>, 3> gf_At_rhs1{
      GF3D2<CCTK_REAL>(layout1, Atxx_rhs), GF3D2<CCTK_REAL>(layout1, Atxy_rhs),
      GF3D2<CCTK_REAL>(layout1, Atxz_rhs), GF3D2<CCTK_REAL>(layout1, Atyy_rhs),
      GF3D2<CCTK_REAL>(layout1, Atyz_rhs), GF3D2<CCTK_REAL>(layout1, Atzz_rhs)};

  const vec<GF3D2<CCTK_REAL>, 3> gf_Gamt_rhs1{
      GF3D2<CCTK_REAL>(layout1, Gamtx_rhs),
      GF3D2<CCTK_REAL>(layout1, Gamty_rhs),
      GF3D2<CCTK_REAL>(layout1, Gamtz_rhs)};

  const GF3D2<CCTK_REAL> gf_Theta_rhs1(layout1, Theta_rhs);

  const GF3D2<CCTK_REAL> gf_alphaG_rhs1(layout1, alphaG_rhs);

  const vec<GF3D2<CCTK_REAL>, 3> gf_betaG_rhs1{
      GF3D2<CCTK_REAL>(layout1, betaGx_rhs),
      GF3D2<CCTK_REAL>(layout1, betaGy_rhs),
      GF3D2<CCTK_REAL>(layout1, betaGz_rhs)};

  //

  typedef simd<CCTK_REAL> vreal;
  typedef simdl<CCTK_REAL> vbool;
  constexpr size_t vsize = tuple_size_v<vreal>;

  const Loop::GridDescBaseDevice grid(cctkGH);

#if 1

#ifdef __CUDACC__
  const nvtxRangeId_t range = nvtxRangeStartA("Z4c_RHS::rhs");
#endif
  noinline([&]() __attribute__((__flatten__, __hot__)) {
    grid.loop_int_device<0, 0, 0, vsize>(
        grid.nghostzones, [=] ARITH_DEVICE(const PointDesc &p) ARITH_INLINE {
          const vbool mask = mask_for_loop_tail<vbool>(p.i, p.imax);
          const GF3D2index index1(layout1, p.I);
          const GF3D5index index0(layout0, p.I);
          const vreal alphaG_value = gf_alphaG0(mask, index0);
          const vreal lapse =
              fmax(vreal(alphaG_floor), vreal(1) + alphaG_value);
          const vreal kappa1_local =
              covariant_z4_damping ? vreal(kappa1) / lapse : vreal(kappa1);

          const CCTK_REAL eta_local =
              eta_at_point(eta_profile_data, p.x, p.y, p.z, veta_width,
                           veta_central, veta_outer);

          // Load and calculate
          const z4c_vars<vreal> vars(
              set_Theta_zero, kappa1_local, kappa2, f_mu_L, f_mu_S,
              eta_local, //
              gf_chi0(mask, index0), gf_dchi0(mask, index0),
              gf_ddchi0(mask, index0), //
              gf_gammat0(mask, index0), gf_dgammat0(mask, index0),
              gf_ddgammat0(mask, index0),                        //
              gf_Kh0(mask, index0), gf_dKh0(mask, index0),       //
              gf_At0(mask, index0), gf_dAt0(mask, index0),       //
              gf_Gamt0(mask, index0), gf_dGamt0(mask, index0),   //
              gf_Theta0(mask, index0), gf_dTheta0(mask, index0), //
              alphaG_value, gf_dalphaG0(mask, index0),
              gf_ddalphaG0(mask, index0), //
              gf_betaG0(mask, index0), gf_dbetaG0(mask, index0),
              gf_ddbetaG0(mask, index0), //
              gf_eTtt1(mask, index1), gf_eTti1(mask, index1),
              gf_eTij1(mask, index1));

          gf_chi_rhs1.store(mask, index1, vars.chi_rhs);
          gf_gammat_rhs1.store(mask, index1, vars.gammat_rhs);
          gf_Kh_rhs1.store(mask, index1, vars.Kh_rhs);
          gf_At_rhs1.store(mask, index1, vars.At_rhs);
          gf_Gamt_rhs1.store(mask, index1, vars.Gamt_rhs);
          gf_Theta_rhs1.store(mask, index1, vars.Theta_rhs);
          gf_alphaG_rhs1.store(mask, index1, vars.alphaG_rhs);
          gf_betaG_rhs1.store(mask, index1, vars.betaG_rhs);
        });
  });
#ifdef __CUDACC__
  nvtxRangeEnd(range);
#endif

#else

  noinline([&]() __attribute__((__flatten__, __hot__)) {
    grid.loop_int_device<0, 0, 0, vsize>(
        grid.nghostzones, [=] ARITH_DEVICE(const PointDesc &p) ARITH_INLINE {
          const vbool mask = mask_for_loop_tail<vbool>(p.i, p.imax);
          const GF3D2index index1(layout1, p.I);
          const GF3D5index index0(layout0, p.I);

          // Load and calculate
          const z4c_vars<vreal> vars(
              kappa1, kappa2, f_mu_L, f_mu_S, eta, //
              gf_chi0(mask, index0), gf_dchi0(mask, index0),
              gf_ddchi0(mask, index0), //
              gf_gammat0(mask, index0), gf_dgammat0(mask, index0),
              gf_ddgammat0(mask, index0),                        //
              gf_Kh0(mask, index0), gf_dKh0(mask, index0),       //
              gf_At0(mask, index0), gf_dAt0(mask, index0),       //
              gf_Gamt0(mask, index0), gf_dGamt0(mask, index0),   //
              gf_Theta0(mask, index0), gf_dTheta0(mask, index0), //
              gf_alphaG0(mask, index0), gf_dalphaG0(mask, index0),
              gf_ddalphaG0(mask, index0), //
              gf_betaG0(mask, index0), gf_dbetaG0(mask, index0),
              gf_ddbetaG0(mask, index0), //
              gf_eTtt1(mask, index1), gf_eTti1(mask, index1),
              gf_eTij1(mask, index1));

          // Store Kh_rhs, At_rhs, Gamt_rhs, Theta_rhs
          gf_Kh_rhs1.store(mask, index1, vars.Kh_rhs);
          gf_At_rhs1.store(mask, index1, vars.At_rhs);
          gf_Gamt_rhs1.store(mask, index1, vars.Gamt_rhs);
          gf_Theta_rhs1.store(mask, index1, vars.Theta_rhs);
        });
  });

  noinline([&]() __attribute__((__flatten__, __hot__)) {
    grid.loop_int_device<0, 0, 0, vsize>(
        grid.nghostzones, [=] ARITH_DEVICE(const PointDesc &p) ARITH_INLINE {
          const vbool mask = mask_for_loop_tail<vbool>(p.i, p.imax);
          const GF3D2index index1(layout1, p.I);
          const GF3D5index index0(layout0, p.I);

          // Load and calculate
          const z4c_vars<vreal> vars(
              kappa1, kappa2, f_mu_L, f_mu_S, eta, //
              gf_chi0(mask, index0), gf_dchi0(mask, index0),
              gf_ddchi0(mask, index0), //
              gf_gammat0(mask, index0), gf_dgammat0(mask, index0),
              gf_ddgammat0(mask, index0),                        //
              gf_Kh0(mask, index0), gf_dKh0(mask, index0),       //
              gf_At0(mask, index0), gf_dAt0(mask, index0),       //
              gf_Gamt0(mask, index0), gf_dGamt0(mask, index0),   //
              gf_Theta0(mask, index0), gf_dTheta0(mask, index0), //
              gf_alphaG0(mask, index0), gf_dalphaG0(mask, index0),
              gf_ddalphaG0(mask, index0), //
              gf_betaG0(mask, index0), gf_dbetaG0(mask, index0),
              gf_ddbetaG0(mask, index0), //
              gf_eTtt1(mask, index1), gf_eTti1(mask, index1),
              gf_eTij1(mask, index1));

          // Store chi_rhs, gammat_rhs, alphaG_rhs, betaG_rhs
          gf_chi_rhs1.store(mask, index1, vars.chi_rhs);
          gf_gammat_rhs1.store(mask, index1, vars.gammat_rhs);
          gf_alphaG_rhs1.store(mask, index1, vars.alphaG_rhs);
          gf_betaG_rhs1.store(mask, index1, vars.betaG_rhs);
        });
  });

#endif

  // Upwind and dissipation terms

  // TODO: Consider fusing the loops to reduce memory bandwidth

  apply_upwind_diss(cctkGH, gf_chi1, gf_betaG1, gf_chi_rhs1);

  for (int a = 0; a < 3; ++a)
    for (int b = a; b < 3; ++b)
      apply_upwind_diss(cctkGH, gf_gammat1(a, b), gf_betaG1,
                        gf_gammat_rhs1(a, b));

  apply_upwind_diss(cctkGH, gf_Kh1, gf_betaG1, gf_Kh_rhs1);

  for (int a = 0; a < 3; ++a)
    for (int b = a; b < 3; ++b)
      apply_upwind_diss(cctkGH, gf_At1(a, b), gf_betaG1, gf_At_rhs1(a, b));

  for (int a = 0; a < 3; ++a)
    apply_upwind_diss(cctkGH, gf_Gamt1(a), gf_betaG1, gf_Gamt_rhs1(a));

  if (!set_Theta_zero)
    apply_upwind_diss(cctkGH, gf_Theta1, gf_betaG1, gf_Theta_rhs1);

  apply_upwind_diss(cctkGH, gf_alphaG1, gf_betaG1, gf_alphaG_rhs1,
                    lapse_advection_coefficient);

  for (int a = 0; a < 3; ++a)
    apply_upwind_diss(cctkGH, gf_betaG1(a), gf_betaG1, gf_betaG_rhs1(a),
                      shift_advection_coefficient);
}

extern "C" void Z4c_Sync(CCTK_ARGUMENTS) {
  // do nothing
}

} // namespace Z4c
