#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameter.h>
#include <cctk_Parameters.h>

namespace {

bool keyword_parameter_equals(const char *const thorn, const char *const name,
                              const char *const expected) {
  int type = -1;
  const void *const value = CCTK_ParameterGet(name, thorn, &type);
  if (!value || type != PARAMETER_KEYWORD)
    return false;

  const char *const keyword = *static_cast<const char *const *>(value);
  return CCTK_Equals(keyword, expected);
}

} // namespace

extern "C" void Z4c_ParamCheck(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS_Z4c_ParamCheck;
  DECLARE_CCTK_PARAMETERS;

  if (!slow_start_lapse)
    return;

  const bool uses_W =
      CCTK_Equals(initial_lapse, "W") ||
      (CCTK_Equals(initial_lapse, "BHClusterX") &&
       keyword_parameter_equals("BHClusterX", "lapse_profile",
                                "precollapsed")) ||
      (CCTK_Equals(initial_lapse, "GRTresnaIDX") &&
       keyword_parameter_equals("GRTresnaIDX", "lapse_profile",
                                "determinant-psi"));

  if (!uses_W)
    CCTK_PARAMWARN(
        "Z4c::slow_start_lapse requires the initial lapse "
        "alpha(t=0)=W=psi^-2. Set ADMBaseX::initial_lapse=\"W\" for "
        "TwoPuncturesX, select BHClusterX::lapse_profile=\"precollapsed\", "
        "or use GRTresnaIDX::lapse_profile=\"determinant-psi\".");
}
