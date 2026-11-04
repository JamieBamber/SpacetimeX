#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>
#include <cctk_Sync.h>

namespace WeylScal4 {

extern "C" void WeylScal4_SyncPsi4(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS_WeylScal4_SyncPsi4;
  DECLARE_CCTK_PARAMETERS;

  // The Psi4 calculation itself uses this cadence guard.  Keeping the
  // synchronization guard identical avoids an otherwise unconditional
  // collective ghost exchange on iterations where Psi4 was not updated.
  // A zero interval disables Psi4, so synchronization must be disabled too;
  // this check also prevents modulo by zero below.
  if (WeylScal4_psi4_calc_4th_calc_every == 0)
    return;
  // CCTK_ANALYSIS is entered for subcycled level batches as well.  Synchrony
  // is required both for a complete Psi4 field and for a meaningful global
  // ghost exchange.
  if (!CarpetX_AllLevelsSynchronized())
    return;
  if (cctk_iteration % WeylScal4_psi4_calc_4th_calc_every !=
      WeylScal4_psi4_calc_4th_calc_offset)
    return;

  const bool trace_this_sync = debug_psi4_trace && CCTK_MyProc(cctkGH) == 0;
  if (trace_this_sync)
    CCTK_VInfo(CCTK_THORNSTRING,
               "Psi4 trace: synchronizing Psi4 groups at iteration=%d",
               cctk_iteration);

  const int groups[] = {
      CCTK_GroupIndex("WeylScal4::Psi4i_group"),
      CCTK_GroupIndex("WeylScal4::Psi4r_group"),
  };
  for (const int group : groups)
    if (group < 0)
      CCTK_VERROR("Could not find a Psi4 grid-function group");

  const int ierr = CCTK_SyncGroupsI(cctkGH, 2, groups);
  if (ierr < 0)
    CCTK_VERROR("Could not synchronize Psi4 grid-function groups: error %d",
                ierr);
  if (trace_this_sync)
    CCTK_VInfo(CCTK_THORNSTRING,
               "Psi4 trace: finished synchronizing Psi4 groups at iteration=%d",
               cctk_iteration);
}

} // namespace WeylScal4
