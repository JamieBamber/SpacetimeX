#include "puncture.hxx"

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameter.h>
#include <cctk_Parameters.h>
#include <util_Table.h>

#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <ctype.h>

namespace PunctureTracker {

static PunctureContainer *g_punctures = nullptr;

static int previous_iteration = 0;

constexpr int max_num_tracked = 100;

static int getCarpetXFinestLevel() {
  int type = 0;
  const void *const value =
      CCTK_ParameterGet("max_num_levels", "CarpetX", &type);
  if (value == nullptr || type != PARAMETER_INT) {
    CCTK_ERROR("Could not query CarpetX::max_num_levels for "
               "PunctureTracker level-cadence tracking. Set "
               "PunctureTracker::tracking_finest_level explicitly.");
  }

  const int maxNumLevels = int(*static_cast<const CCTK_INT *>(value));
  if (maxNumLevels < 1) {
    CCTK_VERROR("Invalid CarpetX::max_num_levels=%d", maxNumLevels);
  }

  return maxNumLevels - 1;
}

static bool shouldTrackThisIteration(const CCTK_INT iteration,
                                     const int trackingLevel,
                                     const int finestLevel) {
  if (trackingLevel < 0) {
    CCTK_ERROR("tracking_level_mode='level' requires tracking_level >= 0");
  }
  if (trackingLevel > finestLevel) {
    CCTK_VERROR("PunctureTracker::tracking_level=%d is finer than finest "
                "tracking level=%d",
                trackingLevel, finestLevel);
  }

  const int levelDifference = finestLevel - trackingLevel;
  if (levelDifference >= int(8 * sizeof(CCTK_INT) - 1)) {
    CCTK_VERROR("PunctureTracker level cadence 2^%d overflows CCTK_INT",
                levelDifference);
  }

  const CCTK_INT stride = CCTK_INT(1) << levelDifference;
  return stride <= 1 || iteration % stride == 0;
}

extern "C" void PunctureTracker_Init(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS_PunctureTracker_Init;
  DECLARE_CCTK_PARAMETERS;

  if (verbose) {
    CCTK_INFO("Initializing PunctureTracker");
  }

  pt_num_tracked[0] = 0;
  pt_num_groups[0] = 0;
  for (int n = 0; n < max_num_tracked; ++n) {
    if (track[n]) {
      pt_loc_t[n] = cctk_time;
      pt_loc_x[n] = initial_x[n];
      pt_loc_y[n] = initial_y[n];
      pt_loc_z[n] = initial_z[n];
      pt_vel_t[n] = cctk_time;
      pt_vel_x[n] = 0.0;
      pt_vel_y[n] = 0.0;
      pt_vel_z[n] = 0.0;
      pt_mass[n] = puncture_mass[n];
      pt_eta_weight[n] = puncture_eta_weight[n];
      ++pt_num_tracked[0];
    } else {
      pt_loc_t[n] = 0.0;
      pt_loc_x[n] = 0.0;
      pt_loc_y[n] = 0.0;
      pt_loc_z[n] = 0.0;
      pt_vel_t[n] = 0.0;
      pt_vel_x[n] = 0.0;
      pt_vel_y[n] = 0.0;
      pt_vel_z[n] = 0.0;
      pt_mass[n] = 0.0;
      pt_eta_weight[n] = 0.0;
    }
    pt_group_membership[n] = -1;
    pt_group_t[n] = 0.0;
    pt_group_x[n] = 0.0;
    pt_group_y[n] = 0.0;
    pt_group_z[n] = 0.0;
    pt_group_mass[n] = 0.0;
    pt_group_eta_weight[n] = 0.0;
  }
}

extern "C" void PunctureTracker_Setup(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS_PunctureTracker_Setup;
  DECLARE_CCTK_PARAMETERS;

  // Initialize PunctureContainer
  if (g_punctures == nullptr) {
    g_punctures = new PunctureContainer();

    for (int n = 0; n < max_num_tracked; ++n) {
      if (track[n]) {
        g_punctures->getTime().push_back(pt_loc_t[n]);
        g_punctures->getLocation()[0].push_back(pt_loc_x[n]);
        g_punctures->getLocation()[1].push_back(pt_loc_y[n]);
        g_punctures->getLocation()[2].push_back(pt_loc_z[n]);
        g_punctures->getVelocity()[0].push_back(pt_vel_x[n]);
        g_punctures->getVelocity()[1].push_back(pt_vel_y[n]);
        g_punctures->getVelocity()[2].push_back(pt_vel_z[n]);
        g_punctures->getMass().push_back(pt_mass[n]);
        g_punctures->getEtaWeight().push_back(pt_eta_weight[n]);
      }
    }
  }

  const int nPunctures = g_punctures->getTime().size();
  g_punctures->getPreviousTime().resize(nPunctures);
  g_punctures->getBeta()[0].resize(nPunctures);
  g_punctures->getBeta()[1].resize(nPunctures);
  g_punctures->getBeta()[2].resize(nPunctures);
  g_punctures->getPreviousBeta()[0].resize(nPunctures);
  g_punctures->getPreviousBeta()[1].resize(nPunctures);
  g_punctures->getPreviousBeta()[2].resize(nPunctures);

  for (int i = 0; i < Loop::dim; ++i) {
    for (int n = 0; n < nPunctures; ++n) {
      g_punctures->getPreviousBeta()[i][n] =
          -g_punctures->getVelocity()[i][n];
    }
  }

  g_punctures->setNumPunctures();
  assert(g_punctures->getNumPunctures() == nPunctures);
  g_punctures->updateGroups(track_mergers, merger_distance_coefficient);

  pt_num_tracked[0] = nPunctures;
  pt_num_groups[0] = CCTK_INT(g_punctures->getGroupMass().size());
  for (int n = 0; n < max_num_tracked; ++n) {
    pt_group_membership[n] = -1;
    pt_group_t[n] = 0.0;
    pt_group_x[n] = 0.0;
    pt_group_y[n] = 0.0;
    pt_group_z[n] = 0.0;
    pt_group_mass[n] = 0.0;
    pt_group_eta_weight[n] = 0.0;
  }
  for (int n = 0; n < nPunctures; ++n) {
    pt_group_membership[n] = g_punctures->getGroupMembership()[n];
  }
  for (int n = 0; n < pt_num_groups[0]; ++n) {
    pt_group_t[n] = cctk_time;
    pt_group_x[n] = g_punctures->getGroupLocation()[0][n];
    pt_group_y[n] = g_punctures->getGroupLocation()[1][n];
    pt_group_z[n] = g_punctures->getGroupLocation()[2][n];
    pt_group_mass[n] = g_punctures->getGroupMass()[n];
    pt_group_eta_weight[n] = g_punctures->getGroupEtaWeight()[n];
  }

  // enabled if refinement regions should follow the punctures
  if (track_boxes) {
    const std::array<std::vector<CCTK_REAL>, Loop::dim> &location =
        g_punctures->getLocation();
    for (int n = 0; n < nPunctures; ++n) {
      CCTK_VINFO("Writing punc coords to box %d.", n);
      position_x[n] = location[0][n];
      position_y[n] = location[1][n];
      position_z[n] = location[2][n];
    }
  }
}

extern "C" void PunctureTracker_Finalize(CCTK_ARGUMENTS) {
  delete g_punctures;
  g_punctures = nullptr;
}

extern "C" void PunctureTracker_Track(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS_PunctureTracker_Track;
  DECLARE_CCTK_PARAMETERS;

#ifndef AMREX_USE_GPU
  assert(!omp_in_parallel());
#endif

  if (g_punctures == nullptr) {
    CCTK_ERROR("PunctureTracker_Track called before PunctureTracker_Setup");
  }

  bool doTrack = true;

  // we can remove this segment when global mode works
  if (cctk_iteration == previous_iteration) {
    doTrack = false;
  } else {
    previous_iteration = cctk_iteration;
  }

  // Do not track while setting up initial data; time interpolation may fail
  if (cctk_iteration == 0) {
    doTrack = false;
  }

  if (doTrack && CCTK_EQUALS(tracking_level_mode, "level")) {
    const int finestLevel =
        tracking_finest_level >= 0 ? int(tracking_finest_level)
                                   : getCarpetXFinestLevel();
    if (!shouldTrackThisIteration(cctk_iteration, int(tracking_level),
                                  finestLevel)) {
      doTrack = false;
    }
  }

  const int nPunctures = g_punctures->getNumPunctures();

  if (doTrack) {
    const std::array<std::vector<CCTK_REAL>, Loop::dim> &location =
        g_punctures->getLocation();

    // Some output
    if (verbose) {
      CCTK_INFO("Tracking punctures...");
      for (int n = 0; n < nPunctures; ++n) {
        CCTK_VINFO("Puncture #%d is at (%g,%g,%g)", n, double(location[0][n]),
                   double(location[1][n]), double(location[2][n]));
      }
    }

    // Manual time level cycling
    g_punctures->updatePreviousTime(CCTK_PASS_CTOC);

    // Interpolate
    g_punctures->interpolate(CCTK_PASS_CTOC);

    if (CCTK_MyProc(cctkGH) == 0) {
      const std::array<std::vector<CCTK_REAL>, Loop::dim> &beta =
          g_punctures->getBeta();

      // More output
      if (verbose) {
        for (int n = 0; n < nPunctures; ++n) {
          CCTK_VINFO("Shift at puncture #%d is at (%g,%g,%g)", n,
                     double(beta[0][n]), double(beta[1][n]),
                     double(beta[2][n]));
        }
      }

      // Check for NaNs and large shift components
      for (int n = 0; n < nPunctures; ++n) {
        const CCTK_REAL norm = std::sqrt(beta[0][n] * beta[0][n] +
                                         beta[1][n] * beta[1][n] +
                                         beta[2][n] * beta[2][n]);

        if (!CCTK_isfinite(norm) || norm > shift_limit) {
          CCTK_VERROR("Shift at puncture #%d is (%g,%g,%g).  This likely "
                      "indicates an error in the simulation.",
                      n, double(beta[0][n]), double(beta[1][n]),
                      double(beta[2][n]));
        }
      }
    }

    // Time evolution
    g_punctures->evolve(CCTK_PASS_CTOC);

    // Broadcast result: 3 components for location, 3 components for velocity
    g_punctures->broadcast(CCTK_PASS_CTOC);
  }

  g_punctures->updateGroups(track_mergers, merger_distance_coefficient);

  const std::array<std::vector<CCTK_REAL>, Loop::dim> &location =
      g_punctures->getLocation();
  const std::array<std::vector<CCTK_REAL>, Loop::dim> &velocity =
      g_punctures->getVelocity();
  const std::vector<CCTK_REAL> &time = g_punctures->getTime();

  // Write to pt_loc_foo and pt_vel_foo
  pt_num_tracked[0] = nPunctures;
  pt_num_groups[0] = CCTK_INT(g_punctures->getGroupMass().size());
  for (int i = 0; i < max_num_tracked; ++i) {
    pt_group_membership[i] = -1;
    pt_group_t[i] = 0.0;
    pt_group_x[i] = 0.0;
    pt_group_y[i] = 0.0;
    pt_group_z[i] = 0.0;
    pt_group_mass[i] = 0.0;
    pt_group_eta_weight[i] = 0.0;
  }
  for (int i = 0; i < nPunctures; ++i) {
    pt_loc_t[i] = time[i];
    pt_loc_x[i] = location[0][i];
    pt_loc_y[i] = location[1][i];
    pt_loc_z[i] = location[2][i];
    pt_vel_t[i] = time[i];
    pt_vel_x[i] = velocity[0][i];
    pt_vel_y[i] = velocity[1][i];
    pt_vel_z[i] = velocity[2][i];
    pt_mass[i] = g_punctures->getMass()[i];
    pt_eta_weight[i] = g_punctures->getEtaWeight()[i];
    pt_group_membership[i] = g_punctures->getGroupMembership()[i];
  }
  for (int i = 0; i < pt_num_groups[0]; ++i) {
    pt_group_t[i] = cctk_time;
    pt_group_x[i] = g_punctures->getGroupLocation()[0][i];
    pt_group_y[i] = g_punctures->getGroupLocation()[1][i];
    pt_group_z[i] = g_punctures->getGroupLocation()[2][i];
    pt_group_mass[i] = g_punctures->getGroupMass()[i];
    pt_group_eta_weight[i] = g_punctures->getGroupEtaWeight()[i];
  }

  if (track_boxes) {
    for (int i = 0; i < nPunctures; ++i) {
      position_x[i] = location[0][i];
      position_y[i] = location[1][i];
      position_z[i] = location[2][i];
    }
  }
}

} // namespace PunctureTracker
