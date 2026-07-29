// announce.cc -- annnounce apparent horizon info to other thorns
// $Header$
//
// <<<access to persistent data>>>
// AHFinderDirect_announce - top-level driver for announce stuff
//

#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <algorithm>

#include "util_Table.h"
#include "cctk.h"
#include "cctk_Arguments.h"
#include "cctk_Parameters.h"
#include "cctk_Functions.h"

#include "config.h"
#include "stdc.h"
#include "../jtutil/util.hh"
#include "../jtutil/array.hh"
#include "../jtutil/cpm_map.hh"
#include "../jtutil/linear_map.hh"
using jtutil::error_exit;

#include "../patch/coords.hh"
#include "../patch/grid.hh"
#include "../patch/fd_grid.hh"
#include "../patch/patch.hh"
#include "../patch/patch_edge.hh"
#include "../patch/patch_interp.hh"
#include "../patch/ghost_zone.hh"
#include "../patch/patch_system.hh"

#include "../elliptic/Jacobian.hh"

#include "../gr/gfns.hh"
#include "../gr/gr.hh"

#include "horizon_sequence.hh"
#include "BH_diagnostics.hh"
#include "driver.hh"

// all the code in this file is inside this namespace
namespace AHFinderDirect
	  {

//******************************************************************************

//
// ***** access to persistent data *****
//
extern struct state state;

//******************************************************************************

//
// This function is called by the Cactus scheduler, to announce any
// desired apparent horizon info to any other thorns that may be interested.
// At present the only info we announce is the centroid position of a
// single selected apparent horizon; if the SetAHCentroid() aliased
// function has been defined then we announce by calling that.
//
extern "C"
  void AHFinderDirect_announce(CCTK_ARGUMENTS)
{
DECLARE_CCTK_ARGUMENTS_AHFinderDirect_announce
DECLARE_CCTK_PARAMETERS

const struct verbose_info& verbose_info = state.verbose_info;

// which horizon to announce?
const int hn = which_horizon_to_announce_centroid;
if (hn == 0)
   then return;						// *** NO-OP RETURN ***

if (! ((hn >= 1) && (hn <= N_horizons)) )
   then CCTK_VWarn(FATAL_ERROR, __LINE__, __FILE__, CCTK_THORNSTRING,
"\n"
"   AHFinderDirect_announce():\n"
"        invalid horizon number %d to announce\n"
"        (valid range is [1,N_horizons=%d])!\n"
		   ,
		   hn, int(N_horizons));			/*NOTREACHED*/

assert(state.AH_data_array[hn] != NULL);
const struct AH_data& AH_data = *state.AH_data_array[hn];

// only try to announce AH info if we've found AHs at this time level
if (! AH_data.search_flag)
   then return;						// *** NO-OP RETURN ***

// did we actually *find* this horizon?
if (! AH_data.found_flag)
   then return;						// *** NO-OP RETURN ***

// is there anyone to announce it to?
if (CCTK_IsFunctionAliased("SetDriftCorrectPosition"))
   then {
	const struct BH_diagnostics& BH_diagnostics = AH_data.BH_diagnostics;
	const CCTK_REAL xx = BH_diagnostics.centroid_x;
	const CCTK_REAL yy = BH_diagnostics.centroid_y;
	const CCTK_REAL zz = BH_diagnostics.centroid_z;
	if (verbose_info.print_physics_details)
	   then CCTK_VInfo(CCTK_THORNSTRING,
			   "horizon %d centroid (%g,%g,%g) --> DriftCorrect",
			   hn, double(xx), double(yy), double(zz));
	SetDriftCorrectPosition(cctkGH, xx, yy, zz);
	}
}

//******************************************************************************

//
// This function is called by the Cactus scheduler, to copy any
// desired apparent horizon info to Cactus variables.
//
extern "C"
  void AHFinderDirect_store(CCTK_ARGUMENTS)
{
DECLARE_CCTK_ARGUMENTS_AHFinderDirect_store
DECLARE_CCTK_PARAMETERS

for (int hn = 1; hn <= N_horizons; ++ hn)
  {

  // Store in spherical surface
  const int sn = sf_IdFromName(which_surface_to_store_info[hn], 
                               which_surface_to_store_info_by_name[hn]);
  if (sn == -1)
    then continue;

  if (sn < 0 || sn >= nsurfaces)
    then CCTK_VWarn(FATAL_ERROR, __LINE__, __FILE__, CCTK_THORNSTRING,
"\n"
"   AHFinderDirect_store():\n"
"        invalid surface number %d for horizon number %d\n"
"        (valid range is [0,nsurfaces-1=%d])!\n"
		   ,
		   sn, hn,
                   int(nsurfaces-1));			/*NOTREACHED*/
  
  const struct AH_data& AH_data = *state.AH_data_array[hn];
  const struct BH_diagnostics& BH_diagnostics = AH_data.BH_diagnostics;
  BH_diagnostics.store(cctkGH, hn, sn);

  }
}

//******************************************************************************

//
// This function is called by the Cactus scheduler, to copy any
// desired apparent horizon info to Cactus variables.
//
extern "C"
  void AHFinderDirect_save(CCTK_ARGUMENTS)
{
DECLARE_CCTK_ARGUMENTS_AHFinderDirect_save
DECLARE_CCTK_PARAMETERS

for (int hn = 1; hn <= N_horizons; ++ hn)
  {

  const struct AH_data& AH_data = *state.AH_data_array[hn];
  const struct BH_diagnostics& BH_diagnostics = AH_data.BH_diagnostics;

  // Save in grid array
  BH_diagnostics.save(cctkGH, hn);

  }
}

//******************************************************************************

//
// This function is called by the Cactus scheduler, to copy any
// desired apparent horizon info from Cactus variables.
//
extern "C"
  void AHFinderDirect_recover(CCTK_ARGUMENTS)
{
DECLARE_CCTK_ARGUMENTS_AHFinderDirect_recover
DECLARE_CCTK_PARAMETERS

for (int hn = 1; hn <= N_horizons; ++ hn)
  {

  struct AH_data& AH_data = *state.AH_data_array[hn];
  struct BH_diagnostics& BH_diagnostics = AH_data.BH_diagnostics;

  // Load from grid array
  BH_diagnostics.load(cctkGH, hn);

  }

// Candidate initial guesses are derived from their recovered parents rather
// than checkpointed separately.  This must be done only after all slots have
// been loaded, since a parent may occupy any earlier or later horizon slot.
for (int candidate_hn = 1; candidate_hn <= N_horizons; ++candidate_hn)
  {
  struct AH_data& candidate = *state.AH_data_array[candidate_hn];
  if (candidate.status != horizon_status__candidate)
     then continue;

  if (candidate.parent_horizons.size() < 2)
     then CCTK_VWarn(
              FATAL_ERROR, __LINE__, __FILE__, CCTK_THORNSTRING,
              "Recovered candidate horizon %d has fewer than two parents",
              candidate_hn);

  fp total_mass = 0.0;
  for (std::vector<int>::const_iterator parent_hn =
           candidate.parent_horizons.begin();
       parent_hn != candidate.parent_horizons.end(); ++parent_hn)
    {
    const struct AH_data& parent = *state.AH_data_array[*parent_hn];
    if (!parent.has_been_found ||
        (parent.status != horizon_status__individual &&
         parent.status != horizon_status__confirmed))
       then CCTK_VWarn(
                FATAL_ERROR, __LINE__, __FILE__, CCTK_THORNSTRING,
                "Recovered candidate horizon %d has ineligible parent %d",
                candidate_hn, *parent_hn);
    total_mass += parent.mass;
    }
  if (!isfinite(total_mass) || total_mass <= 0.0)
     then CCTK_VWarn(
              FATAL_ERROR, __LINE__, __FILE__, CCTK_THORNSTRING,
              "Recovered candidate horizon %d has invalid parent mass sum %g",
              candidate_hn, double(total_mass));

  fp center_x = 0.0;
  fp center_y = 0.0;
  fp center_z = 0.0;
  if (candidate.candidate_method ==
      candidate_discovery_method__method1)
     then {
	for (std::vector<int>::const_iterator parent_hn =
	         candidate.parent_horizons.begin();
	     parent_hn != candidate.parent_horizons.end(); ++parent_hn)
	  {
	  const struct AH_data& parent = *state.AH_data_array[*parent_hn];
	  center_x += parent.mass * parent.BH_diagnostics.centroid_x;
	  center_y += parent.mass * parent.BH_diagnostics.centroid_y;
	  center_z += parent.mass * parent.BH_diagnostics.centroid_z;
	  }
	center_x /= total_mass;
	center_y /= total_mass;
	center_z /= total_mass;
	}
  else if (candidate.candidate_method ==
           candidate_discovery_method__method2)
     then {
	const struct BH_diagnostics& first =
	    state.AH_data_array[candidate.parent_horizons[0]]->BH_diagnostics;
	fp min_x = first.origin_x - first.max_radius;
	fp max_x = first.origin_x + first.max_radius;
	fp min_y = first.origin_y - first.max_radius;
	fp max_y = first.origin_y + first.max_radius;
	fp min_z = first.origin_z - first.max_radius;
	fp max_z = first.origin_z + first.max_radius;
	for (std::vector<int>::size_type i = 1;
	     i < candidate.parent_horizons.size(); ++i)
	  {
	  const struct BH_diagnostics& parent_diagnostics =
	      state.AH_data_array[candidate.parent_horizons[i]]->BH_diagnostics;
	  min_x = std::min(min_x, parent_diagnostics.origin_x -
	                         parent_diagnostics.max_radius);
	  max_x = std::max(max_x, parent_diagnostics.origin_x +
	                         parent_diagnostics.max_radius);
	  min_y = std::min(min_y, parent_diagnostics.origin_y -
	                         parent_diagnostics.max_radius);
	  max_y = std::max(max_y, parent_diagnostics.origin_y +
	                         parent_diagnostics.max_radius);
	  min_z = std::min(min_z, parent_diagnostics.origin_z -
	                         parent_diagnostics.max_radius);
	  max_z = std::max(max_z, parent_diagnostics.origin_z +
	                         parent_diagnostics.max_radius);
	  }
	center_x = 0.5 * (min_x + max_x);
	center_y = 0.5 * (min_y + max_y);
	center_z = 0.5 * (min_z + max_z);
	}
  else CCTK_VWarn(
           FATAL_ERROR, __LINE__, __FILE__, CCTK_THORNSTRING,
           "Recovered candidate horizon %d has invalid discovery method %d",
           candidate_hn, int(candidate.candidate_method));

  const fp candidate_radius = merger_pre_factor * total_mass;
  if (!isfinite(center_x) || !isfinite(center_y) ||
      !isfinite(center_z) || !isfinite(candidate_radius) ||
      candidate_radius <= 0.0)
     then CCTK_VWarn(
              FATAL_ERROR, __LINE__, __FILE__, CCTK_THORNSTRING,
              "Could not reconstruct initial sphere for candidate horizon %d",
              candidate_hn);

  patch_system& ps = *candidate.ps_ptr;
  ps.origin_x(center_x);
  ps.origin_y(center_y);
  ps.origin_z(center_z);
  candidate.BH_diagnostics.origin_x = center_x;
  candidate.BH_diagnostics.origin_y = center_y;
  candidate.BH_diagnostics.origin_z = center_z;
  candidate.initial_guess_info.method = initial_guess__coord_sphere;
  candidate.initial_guess_info.reset_horizon_after_not_finding = true;
  candidate.initial_guess_info.coord_sphere_info.x_center = center_x;
  candidate.initial_guess_info.coord_sphere_info.y_center = center_y;
  candidate.initial_guess_info.coord_sphere_info.z_center = center_z;
  candidate.initial_guess_info.coord_sphere_info.radius = candidate_radius;
  candidate.mass = total_mass;
  }
}

//******************************************************************************

	  }	// namespace AHFinderDirect
