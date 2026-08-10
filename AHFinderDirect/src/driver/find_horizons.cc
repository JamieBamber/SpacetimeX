// find_horizons.cc -- top level driver for finding apparent horizons
// $Header$
//
// <<<access to persistent data>>>
// <<<prototypes for functions local to this file>>>
// AHFinderDirect_find_horizons - top-level driver to find apparent horizons
///
/// find_horizon - find a horizon
/// do_evaluate_expansions
/// do_test_expansion_Jacobian
///

#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <loop_device.hxx>
#include <algorithm>
#include <array>
#include <functional>
#include <string>
#include <vector>

#include "util_Table.h"
#include "cctk.h"
#include "cctk_Arguments.h"
#include "cctk_Parameters.h"

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
// ***** prototypes for functions local to this file
//
namespace {
void do_evaluate_expansions(int my_proc, int N_horizons,
			    horizon_sequence& hs,
			       struct AH_data* const AH_data_array[],
			    const struct cactus_grid_info& cgi,
			    const struct geometry_info& gi,
			    const struct IO_info& IO_info,
			    const struct error_info& error_info,
			    const struct verbose_info& verbose_info,
			    int timer_handle);
void do_test_expansion_Jacobians(int my_proc, int N_horizons,
				 struct AH_data* const AH_data_array[],
				 const struct cactus_grid_info& cgi,
				 const struct geometry_info& gi,
				       struct Jacobian_info& Jac_info,
				 bool test_all_Jacobian_compute_methods,
				 const struct IO_info& IO_info,
				 const struct error_info& error_info,
				 const struct verbose_info& verbose_info,
				 int timer_handle);
void promote_candidate_slot(int candidate_hn);
void write_merger_event(CCTK_ARGUMENTS, int daughter_hn);
void discover_method1_mergers(CCTK_ARGUMENTS);
void discover_method2_mergers(CCTK_ARGUMENTS);
void release_merger_candidates(CCTK_ARGUMENTS);
void set_search_origins_from_punctures(CCTK_ARGUMENTS);
	  }

//******************************************************************************

namespace {
struct origin_estimate {
  fp x;
  fp y;
  fp z;
  fp mass;
};

bool finite_origin(const fp x, const fp y, const fp z)
{
  return std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
}

// Estimate a horizon's current position.  A horizon found on the previous
// search is the best estimate while it remains active.  Otherwise descend its
// merger tree until the live punctures underlying the horizon are reached.
bool hierarchical_origin(const int hn, const CCTK_INT ntracked,
                         const CCTK_REAL* const puncture_x,
                         const CCTK_REAL* const puncture_y,
                         const CCTK_REAL* const puncture_z,
                         std::vector<bool>& visiting,
                         origin_estimate& estimate)
{
  if (hn < 1 || hn > state.N_horizons || visiting[hn]) {
    return false;
  }

  const struct AH_data& horizon = *state.AH_data_array[hn];
  if (horizon.status == horizon_status__individual) {
    const int puncture = hn - 1;
    if (puncture >= ntracked ||
        !finite_origin(puncture_x[puncture], puncture_y[puncture],
                       puncture_z[puncture])) {
      return false;
    }
    estimate.x = puncture_x[puncture];
    estimate.y = puncture_y[puncture];
    estimate.z = puncture_z[puncture];
    estimate.mass = horizon.mass;
    return estimate.mass > 0.0 && std::isfinite(estimate.mass);
  }

  if (horizon.parent_horizons.empty()) {
    return false;
  }

  visiting[hn] = true;
  fp weighted_x = 0.0;
  fp weighted_y = 0.0;
  fp weighted_z = 0.0;
  fp total_mass = 0.0;
  for (std::vector<int>::const_iterator parent_hn =
           horizon.parent_horizons.begin();
       parent_hn != horizon.parent_horizons.end(); ++parent_hn) {
    if (*parent_hn < 1 || *parent_hn > state.N_horizons) {
      visiting[hn] = false;
      return false;
    }

    const struct AH_data& parent = *state.AH_data_array[*parent_hn];
    origin_estimate parent_estimate;
    const fp cx = parent.BH_diagnostics.centroid_x;
    const fp cy = parent.BH_diagnostics.centroid_y;
    const fp cz = parent.BH_diagnostics.centroid_z;
    if (parent.search_flag && parent.found_flag && parent.has_been_found &&
        parent.mass > 0.0 && std::isfinite(parent.mass) &&
        finite_origin(cx, cy, cz)) {
      parent_estimate = {cx, cy, cz, parent.mass};
    } else if (!hierarchical_origin(*parent_hn, ntracked, puncture_x,
                                    puncture_y, puncture_z, visiting,
                                    parent_estimate)) {
      visiting[hn] = false;
      return false;
    }

    weighted_x += parent_estimate.mass * parent_estimate.x;
    weighted_y += parent_estimate.mass * parent_estimate.y;
    weighted_z += parent_estimate.mass * parent_estimate.z;
    total_mass += parent_estimate.mass;
  }
  visiting[hn] = false;

  if (!(total_mass > 0.0) || !std::isfinite(total_mass)) {
    return false;
  }
  estimate = {weighted_x / total_mass, weighted_y / total_mass,
              weighted_z / total_mass, total_mass};
  return finite_origin(estimate.x, estimate.y, estimate.z);
}

void set_initial_guess_center(struct initial_guess_info& guess,
                              const fp x, const fp y, const fp z)
{
  guess.Kerr_Kerr_info.x_posn = x;
  guess.Kerr_Kerr_info.y_posn = y;
  guess.Kerr_Kerr_info.z_posn = z;
  guess.Kerr_KerrSchild_info.x_posn = x;
  guess.Kerr_KerrSchild_info.y_posn = y;
  guess.Kerr_KerrSchild_info.z_posn = z;
  guess.coord_sphere_info.x_center = x;
  guess.coord_sphere_info.y_center = y;
  guess.coord_sphere_info.z_center = z;
  guess.coord_ellipsoid_info.x_center = x;
  guess.coord_ellipsoid_info.y_center = y;
  guess.coord_ellipsoid_info.z_center = z;
}

void set_search_origins_from_punctures(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_ARGUMENTS_AHFinderDirect_find_horizons;

  if (pt_num_tracked == NULL || pt_loc_x == NULL || pt_loc_y == NULL ||
      pt_loc_z == NULL) {
    CCTK_VWarn(FATAL_ERROR, __LINE__, __FILE__, CCTK_THORNSTRING,
               "AH_set_origins_to_punctures requires initialized "
               "PunctureTracker variables");
  }
  const CCTK_INT ntracked = pt_num_tracked[0];

  for (int hn = 1; hn <= state.N_horizons; ++hn) {
    struct AH_data& horizon = *state.AH_data_array[hn];
    if (!horizon.search_flag) {
      continue;
    }

    std::vector<bool> visiting(state.N_horizons + 1, false);
    origin_estimate estimate;
    if (!hierarchical_origin(hn, ntracked, pt_loc_x, pt_loc_y, pt_loc_z,
                             visiting, estimate)) {
      CCTK_VWarn(FATAL_ERROR, __LINE__, __FILE__, CCTK_THORNSTRING,
                 "Cannot determine a puncture-based search origin for "
                 "horizon %d (status=%d, tracked punctures=%d)",
                 hn, int(horizon.status), int(ntracked));
    }

    horizon.ps_ptr->origin_x(estimate.x);
    horizon.ps_ptr->origin_y(estimate.y);
    horizon.ps_ptr->origin_z(estimate.z);
    set_initial_guess_center(horizon.initial_guess_info,
                             estimate.x, estimate.y, estimate.z);
    if (state.my_proc == 0 && state.verbose_info.print_physics_details) {
      CCTK_VInfo(CCTK_THORNSTRING,
                 "Horizon %d search origin set from puncture hierarchy to "
                 "(%.17g,%.17g,%.17g)",
                 hn, double(estimate.x), double(estimate.y),
                 double(estimate.z));
    }
  }
}
}

//******************************************************************************

namespace {
void promote_candidate_slot(const int candidate_hn)
{
  struct AH_data& candidate = *state.AH_data_array[candidate_hn];

  if (candidate.status != horizon_status__candidate ||
      !candidate.found_flag || candidate.has_been_found) {
    CCTK_VWarn(FATAL_ERROR, __LINE__, __FILE__, CCTK_THORNSTRING,
               "Cannot promote horizon %d: status=%d, found=%d, "
               "has_been_found=%d",
               candidate_hn, int(candidate.status),
               int(candidate.found_flag), int(candidate.has_been_found));
  }
  if (candidate.parent_horizons.size() < 2) {
    CCTK_VWarn(FATAL_ERROR, __LINE__, __FILE__, CCTK_THORNSTRING,
               "Cannot promote candidate horizon %d with fewer than two parents",
               candidate_hn);
  }

  // Validate the complete relationship before changing any slot, so a bad
  // candidate cannot leave a partially updated merger tree.
  for (std::vector<int>::const_iterator parent_hn =
           candidate.parent_horizons.begin();
       parent_hn != candidate.parent_horizons.end(); ++parent_hn) {
    if (*parent_hn < 1 || *parent_hn > state.N_horizons ||
        *parent_hn == candidate_hn) {
      CCTK_VWarn(FATAL_ERROR, __LINE__, __FILE__, CCTK_THORNSTRING,
                 "Candidate horizon %d has invalid parent horizon %d",
                 candidate_hn, *parent_hn);
    }
    const struct AH_data& parent = *state.AH_data_array[*parent_hn];
    if (!parent.has_been_found ||
        (parent.status != horizon_status__individual &&
         parent.status != horizon_status__confirmed) ||
        parent.inside_confirmed_merger) {
      CCTK_VWarn(FATAL_ERROR, __LINE__, __FILE__, CCTK_THORNSTRING,
                 "Candidate horizon %d has ineligible parent horizon %d "
                 "(status=%d, has_been_found=%d, inside_merger=%d)",
                 candidate_hn, *parent_hn, int(parent.status),
                 int(parent.has_been_found),
                 int(parent.inside_confirmed_merger));
    }
  }

  for (std::vector<int>::const_iterator parent_hn =
           candidate.parent_horizons.begin();
       parent_hn != candidate.parent_horizons.end(); ++parent_hn) {
    state.AH_data_array[*parent_hn]->inside_confirmed_merger = true;
  }

  candidate.status = horizon_status__confirmed;
  candidate.has_been_found = true;
  candidate.inside_confirmed_merger = false;
  candidate.initial_find_flag = false;
  candidate.really_initial_find_flag = false;
  candidate.candidate_inactive_checks = 0;

  if (state.my_proc == 0) {
    CCTK_VInfo(CCTK_THORNSTRING,
               "Promoted candidate horizon %d to a confirmed horizon",
               candidate_hn);
  }
}
}

//******************************************************************************

// Record a merger once its candidate common horizon has converged.  Each
// record identifies when the merger was found, the newly confirmed daughter
// horizon, the discovery method, and the daughter horizon's parents.
namespace {
void write_merger_event(CCTK_ARGUMENTS, const int daughter_hn)
{
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;

  struct AH_data& daughter = *state.AH_data_array[daughter_hn];
  if (daughter.status != horizon_status__confirmed ||
      !daughter.has_been_found ||
      daughter.parent_horizons.size() < 2) {
    CCTK_VWarn(FATAL_ERROR, __LINE__, __FILE__, CCTK_THORNSTRING,
               "Cannot write merger event for invalid daughter horizon %d",
               daughter_hn);
  }
  if (daughter.merger_event_written) {
    return;
  }

  if (state.my_proc == 0) {
    const char* directory = state.IO_info.BH_diagnostics_directory;
    const int directory_status =
        CCTK_CreateDirectory(IO_info::default_directory_permission, directory);
    if (directory_status < 0) {
      CCTK_VWarn(FATAL_ERROR, __LINE__, __FILE__, CCTK_THORNSTRING,
                 "Error %d creating merger-event output directory \"%s\"",
                 directory_status, directory);
    }

    char file_name[IO_info::file_name_buffer_size];
    snprintf(file_name, IO_info::file_name_buffer_size, "%s/%s",
             directory, merger_event_file_name);

    if (!state.merger_event_file_initialized) {
      const bool truncate_file = IO_TruncateOutputFiles(cctkGH) == 1;
      FILE* initialize_file = NULL;
      if (truncate_file) {
        initialize_file = fopen(file_name, "w");
      } else {
        FILE* existing_file = fopen(file_name, "r");
        if (existing_file != NULL) {
          fclose(existing_file);
        } else {
          initialize_file = fopen(file_name, "w");
        }
      }
      if (initialize_file != NULL) {
        fprintf(initialize_file,
                "# %-10s %-24s %-10s %-18s %-10s %s\n",
                "iteration", "time", "daughter", "discovery_method",
                "nparents", "parent_horizons...");
        if (fclose(initialize_file) != 0) {
          CCTK_VWarn(FATAL_ERROR, __LINE__, __FILE__, CCTK_THORNSTRING,
                     "Error closing merger-event file \"%s\"", file_name);
        }
      } else if (truncate_file) {
        CCTK_VWarn(FATAL_ERROR, __LINE__, __FILE__, CCTK_THORNSTRING,
                   "Cannot create merger-event file \"%s\"", file_name);
      }
      state.merger_event_file_initialized = true;
    }

    // The checkpointed flag prevents normal duplicate writes.  Checking the
    // file as well closes the small recovery window in which the event was
    // written after the checkpoint from which this run resumed.
    bool daughter_already_recorded = false;
    FILE* read_file = fopen(file_name, "r");
    if (read_file != NULL) {
      char line[4096];
      while (fgets(line, sizeof(line), read_file) != NULL) {
        int saved_iteration = 0;
        double saved_time = 0.0;
        int saved_daughter = 0;
        if (sscanf(line, "%d %lf %d",
                   &saved_iteration, &saved_time, &saved_daughter) == 3 &&
            saved_daughter == daughter_hn) {
          daughter_already_recorded = true;
          break;
        }
      }
      fclose(read_file);
    }

    if (!daughter_already_recorded) {
      FILE* append_file = fopen(file_name, "a");
      if (append_file == NULL) {
        CCTK_VWarn(FATAL_ERROR, __LINE__, __FILE__, CCTK_THORNSTRING,
                   "Cannot append to merger-event file \"%s\"", file_name);
      }
      fprintf(append_file, "  %-10d %-24.17g %-10d %-18d %-10d",
              int(cctk_iteration), double(cctk_time), daughter_hn,
              int(daughter.candidate_method),
              int(daughter.parent_horizons.size()));
      for (std::vector<int>::const_iterator parent_hn =
               daughter.parent_horizons.begin();
           parent_hn != daughter.parent_horizons.end(); ++parent_hn) {
        fprintf(append_file, " %d", *parent_hn);
      }
      fprintf(append_file, "\n");
      if (fclose(append_file) != 0) {
        CCTK_VWarn(FATAL_ERROR, __LINE__, __FILE__, CCTK_THORNSTRING,
                   "Error closing merger-event file \"%s\"", file_name);
      }
    }
  }

  // All ranks update the replicated lifecycle metadata.  The file itself is
  // touched only by rank zero.
  daughter.merger_event_written = true;
}
}

//******************************************************************************

namespace {
std::string horizon_group_string(const std::vector<int>& group)
{
  std::string result = "{";
  for (std::vector<int>::size_type i = 0; i < group.size(); ++i) {
    if (i != 0) {
      result += ",";
    }
    result += std::to_string(group[i]);
  }
  result += "}";
  return result;
}

void print_all_candidate_groups(const char* const context)
{
  if (state.my_proc != 0) {
    return;
  }
  int count = 0;
  for (int hn = 1; hn <= state.my_hs->N_horizons(); ++hn) {
    const struct AH_data& candidate = *state.AH_data_array[hn];
    if (candidate.status == horizon_status__candidate) {
      ++count;
      const std::string group =
          horizon_group_string(candidate.parent_horizons);
      CCTK_VInfo(CCTK_THORNSTRING,
                 "%s: candidate horizon %d has parent group %s",
                 context, hn, group.c_str());
    }
  }
  if (count == 0) {
    CCTK_VInfo(CCTK_THORNSTRING, "%s: no active candidate horizon groups",
               context);
  } else {
    CCTK_VInfo(CCTK_THORNSTRING,
               "%s: %d active candidate horizon group%s in total",
               context, count, count == 1 ? "" : "s");
  }
}

void discover_method1_mergers(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;

  if (!discover_mergers) {
    return;
  }

  if (state.my_proc == 0) {
    CCTK_VInfo(CCTK_THORNSTRING,
               "Merger discovery Method #1 starting at iteration %d",
               int(cctk_iteration));
  }

  // The eligible horizons are the current outer frontier of the merger tree:
  // real horizons which have not already been enclosed by a confirmed child.
  std::vector<int> eligible;
  for (int hn = 1; hn <= N_horizons; ++hn) {
    const struct AH_data& AH_data = *state.AH_data_array[hn];
    if (AH_data.has_been_found &&
        !AH_data.inside_confirmed_merger &&
        (AH_data.status == horizon_status__individual ||
         AH_data.status == horizon_status__confirmed)) {
      eligible.push_back(hn);
      if (state.my_proc == 0) {
        CCTK_VInfo(
            CCTK_THORNSTRING,
            "Method #1 eligible AH %d: has_been_found=%s, status=%d, "
            "centroid=(%.17g,%.17g,%.17g), proxy_mass=%.17g",
            hn, AH_data.has_been_found ? "yes" : "no", int(AH_data.status),
            double(AH_data.BH_diagnostics.centroid_x),
            double(AH_data.BH_diagnostics.centroid_y),
            double(AH_data.BH_diagnostics.centroid_z),
            double(AH_data.mass));
      }
    }
  }
  const int neligible = int(eligible.size());
  std::vector<std::vector<bool> > adjacent(
      neligible, std::vector<bool>(neligible, false));
  for (int i = 0; i < neligible; ++i) {
    const struct AH_data& AH_i = *state.AH_data_array[eligible[i]];
    for (int j = i + 1; j < neligible; ++j) {
      const struct AH_data& AH_j = *state.AH_data_array[eligible[j]];
      const fp dx = AH_i.BH_diagnostics.centroid_x -
                    AH_j.BH_diagnostics.centroid_x;
      const fp dy = AH_i.BH_diagnostics.centroid_y -
                    AH_j.BH_diagnostics.centroid_y;
      const fp dz = AH_i.BH_diagnostics.centroid_z -
                    AH_j.BH_diagnostics.centroid_z;
      const fp distance = sqrt(dx*dx + dy*dy + dz*dz);
      const fp min_distance =
          4.0 * merger_search_factor * (AH_i.mass + AH_j.mass);
      if (state.my_proc == 0) {
        CCTK_VInfo(CCTK_THORNSTRING,
                   "Method #1 pair (%d,%d): distance=%.17g, "
                   "min_distance=%.17g, nearby=%s",
                   eligible[i], eligible[j], double(distance),
                   double(min_distance),
                   distance < min_distance ? "yes" : "no");
      }
      adjacent[i][j] = adjacent[j][i] = distance < min_distance;
    }
  }

  std::vector<std::vector<int> > proposed_groups;
  if (merger_allow_group_mergers) {
    // Method #1 as used in GRChombo: each connected component of the
    // pair-proximity graph is one proposed merger group.
    std::vector<bool> visited(neligible, false);
    for (int seed = 0; seed < neligible; ++seed) {
      if (visited[seed]) {
        continue;
      }
      std::vector<int> component_indices(1, seed);
      visited[seed] = true;
      for (std::vector<int>::size_type next = 0;
           next < component_indices.size(); ++next) {
        const int i = component_indices[next];
        for (int j = 0; j < neligible; ++j) {
          if (adjacent[i][j] && !visited[j]) {
            visited[j] = true;
            component_indices.push_back(j);
          }
        }
      }
      if (component_indices.size() >= 2) {
        std::vector<int> group;
        for (std::vector<int>::const_iterator index =
                 component_indices.begin();
             index != component_indices.end(); ++index) {
          group.push_back(eligible[*index]);
        }
        const std::vector<int> canonical_group =
            canonical_parent_group(group);
        proposed_groups.push_back(canonical_group);
        if (state.my_proc == 0) {
          const std::string group_text =
              horizon_group_string(canonical_group);
          CCTK_VInfo(CCTK_THORNSTRING,
                     "Method #1 found new proposed AH group %s",
                     group_text.c_str());
        }
      }
    }
  } else {
    // Pair-only mode keeps every qualifying edge as its own candidate.
    for (int i = 0; i < neligible; ++i) {
      for (int j = i + 1; j < neligible; ++j) {
        if (adjacent[i][j]) {
          std::vector<int> pair;
          pair.push_back(eligible[i]);
          pair.push_back(eligible[j]);
          const std::vector<int> canonical_group =
              canonical_parent_group(pair);
          proposed_groups.push_back(canonical_group);
          if (state.my_proc == 0) {
            const std::string group_text =
                horizon_group_string(canonical_group);
            CCTK_VInfo(CCTK_THORNSTRING,
                       "Method #1 found new proposed AH group %s",
                       group_text.c_str());
          }
        }
      }
    }
  }

  int active_candidates = 0;
  std::vector<std::vector<int> > created_groups;
  for (int hn = 1; hn <= N_horizons; ++hn) {
    if (state.AH_data_array[hn]->status == horizon_status__candidate) {
      ++active_candidates;
    }
  }

  for (std::vector<std::vector<int> >::const_iterator group =
           proposed_groups.begin();
       group != proposed_groups.end(); ++group) {
    if (find_horizon_with_parent_group(*group) != 0) {
      continue;
    }
    if (active_candidates >= max_active_merger_candidates) {
      if (state.my_proc == 0) {
        CCTK_VWarn(CCTK_WARN_ALERT, __LINE__, __FILE__, CCTK_THORNSTRING,
                   "Not creating another Method #1 merger candidate: "
                   "max_active_merger_candidates=%d",
                   int(max_active_merger_candidates));
      }
      break;
    }

    const int candidate_hn = find_unused_horizon_slot();
    if (candidate_hn == 0) {
      if (state.my_proc == 0) {
        CCTK_VWarn(CCTK_WARN_ALERT, __LINE__, __FILE__, CCTK_THORNSTRING,
                   "No unused AHFinderDirect slot remains for a new "
                   "Method #1 merger candidate");
      }
      break;
    }

    fp total_mass = 0.0;
    fp center_x = 0.0;
    fp center_y = 0.0;
    fp center_z = 0.0;
    for (std::vector<int>::const_iterator parent_hn = group->begin();
         parent_hn != group->end(); ++parent_hn) {
      const struct AH_data& parent = *state.AH_data_array[*parent_hn];
      total_mass += parent.mass;
      center_x += parent.mass * parent.BH_diagnostics.centroid_x;
      center_y += parent.mass * parent.BH_diagnostics.centroid_y;
      center_z += parent.mass * parent.BH_diagnostics.centroid_z;
    }
    if (!isfinite(total_mass) || total_mass <= 0.0) {
      CCTK_VWarn(FATAL_ERROR, __LINE__, __FILE__, CCTK_THORNSTRING,
                 "Method #1 proposed a merger group with invalid total "
                 "proxy mass %g",
                 double(total_mass));
    }
    center_x /= total_mass;
    center_y /= total_mass;
    center_z /= total_mass;
    const fp candidate_radius = merger_pre_factor * total_mass;

    initialize_candidate_slot(
        CCTK_PASS_CTOC, candidate_hn, *group,
        center_x, center_y, center_z, candidate_radius,
        candidate_discovery_method__method1);
    // Discovery is performed only on a horizon-finder iteration.  Make the
    // newly initialized candidate participate in the Newton solve later in
    // this same call, independently of the timing parameters attached to its
    // previously unused slot.
    state.AH_data_array[candidate_hn]->search_flag = true;
    ++active_candidates;
    created_groups.push_back(*group);

    if (state.my_proc == 0) {
      CCTK_VInfo(CCTK_THORNSTRING,
                 "Created Method #1 merger candidate horizon %d with "
                 "%d parents, proxy mass %g, and initial radius %g",
                 candidate_hn, int(group->size()), double(total_mass),
                 double(candidate_radius));
    }
  }

  if (state.my_proc == 0) {
    CCTK_VInfo(CCTK_THORNSTRING,
               "Method #1 pass summary: %d eligible horizon%s, "
               "%d proposed group%s, %d new candidate%s created",
               neligible, neligible == 1 ? "" : "s",
               int(proposed_groups.size()),
               proposed_groups.size() == 1 ? "" : "s",
               int(created_groups.size()),
               created_groups.size() == 1 ? "" : "s");
    for (std::vector<std::vector<int> >::const_iterator group =
             created_groups.begin(); group != created_groups.end(); ++group) {
      const std::string group_text = horizon_group_string(*group);
      CCTK_VInfo(CCTK_THORNSTRING,
                 "Method #1 new candidate parent group %s",
                 group_text.c_str());
    }
  }
  print_all_candidate_groups("After Method #1");
}
}

//******************************************************************************

namespace {
void discover_method2_mergers(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;

  if (!discover_mergers || !merger_search_compact_groups) {
    return;
  }

  if (state.my_proc == 0) {
    CCTK_VInfo(CCTK_THORNSTRING,
               "Merger discovery Method #2 starting at iteration %d",
               int(cctk_iteration));
  }

  std::vector<int> eligible;
  for (int hn = 1; hn <= N_horizons; ++hn) {
    const struct AH_data& AH_data = *state.AH_data_array[hn];
    if (AH_data.has_been_found &&
        !AH_data.inside_confirmed_merger &&
        (AH_data.status == horizon_status__individual ||
         AH_data.status == horizon_status__confirmed)) {
      eligible.push_back(hn);
    }
  }
  if (eligible.size() < 3) {
    return;
  }

  int active_candidates = 0;
  for (int hn = 1; hn <= N_horizons; ++hn) {
    if (state.AH_data_array[hn]->status == horizon_status__candidate) {
      ++active_candidates;
    }
  }

  bool stop_search = active_candidates >= max_active_merger_candidates;
  std::vector<int> group;
  const int largest_group =
      std::min(int(eligible.size()), int(merger_compact_max_group_size));

  // Enumerate larger compact groups first: Method #2 is intended to retain
  // the cluster-scale candidates which Method #1's local pair graph may not
  // propose.  Different qualifying subsets may still coexist until the
  // configured active-candidate limit is reached.
  for (int group_size = largest_group;
       group_size >= 3 && !stop_search; --group_size) {
    group.clear();
    std::function<void(int, int)> enumerate =
        [&](const int begin, const int remaining) {
      if (stop_search) {
        return;
      }
      if (remaining > 0) {
        const int last = int(eligible.size()) - remaining;
        for (int i = begin; i <= last && !stop_search; ++i) {
          group.push_back(eligible[i]);
          enumerate(i + 1, remaining - 1);
          group.pop_back();
        }
        return;
      }

      if (find_horizon_with_parent_group(group) != 0) {
        return;
      }

      fp total_mass = 0.0;
      for (std::vector<int>::const_iterator parent_hn = group.begin();
           parent_hn != group.end(); ++parent_hn) {
        total_mass += state.AH_data_array[*parent_hn]->mass;
      }
      if (!isfinite(total_mass) || total_mass <= 0.0) {
        CCTK_VWarn(FATAL_ERROR, __LINE__, __FILE__, CCTK_THORNSTRING,
                   "Method #2 proposed a group with invalid total proxy "
                   "mass %g",
                   double(total_mass));
      }

      const fp pair_limit =
          4.0 * merger_compact_search_factor * total_mass;
      for (std::vector<int>::size_type i = 0; i < group.size(); ++i) {
        const struct BH_diagnostics& di =
            state.AH_data_array[group[i]]->BH_diagnostics;
        for (std::vector<int>::size_type j = i + 1;
             j < group.size(); ++j) {
          const struct BH_diagnostics& dj =
              state.AH_data_array[group[j]]->BH_diagnostics;
          const fp dx = di.centroid_x - dj.centroid_x;
          const fp dy = di.centroid_y - dj.centroid_y;
          const fp dz = di.centroid_z - dj.centroid_z;
          if (sqrt(dx*dx + dy*dy + dz*dz) > pair_limit) {
            return;
          }
        }
      }

      // Treat each horizon as a ball about its patch origin with radius
      // max_radius.  The midpoint of their combined axis-aligned bounding box
      // supplies a deterministic trial center; max(|center-origin_i| +
      // max_radius_i) is a conservative sphere enclosing all those balls.
      fp min_x = 0.0, max_x = 0.0;
      fp min_y = 0.0, max_y = 0.0;
      fp min_z = 0.0, max_z = 0.0;
      for (std::vector<int>::size_type i = 0; i < group.size(); ++i) {
        const struct BH_diagnostics& d =
            state.AH_data_array[group[i]]->BH_diagnostics;
        if (!isfinite(d.max_radius) || d.max_radius < 0.0) {
          CCTK_VWarn(FATAL_ERROR, __LINE__, __FILE__, CCTK_THORNSTRING,
                     "Method #2 parent horizon %d has invalid max_radius %g",
                     group[i], double(d.max_radius));
        }
        if (i == 0) {
          min_x = d.origin_x - d.max_radius;
          max_x = d.origin_x + d.max_radius;
          min_y = d.origin_y - d.max_radius;
          max_y = d.origin_y + d.max_radius;
          min_z = d.origin_z - d.max_radius;
          max_z = d.origin_z + d.max_radius;
        } else {
          min_x = std::min(min_x, d.origin_x - d.max_radius);
          max_x = std::max(max_x, d.origin_x + d.max_radius);
          min_y = std::min(min_y, d.origin_y - d.max_radius);
          max_y = std::max(max_y, d.origin_y + d.max_radius);
          min_z = std::min(min_z, d.origin_z - d.max_radius);
          max_z = std::max(max_z, d.origin_z + d.max_radius);
        }
      }
      const fp center_x = 0.5 * (min_x + max_x);
      const fp center_y = 0.5 * (min_y + max_y);
      const fp center_z = 0.5 * (min_z + max_z);
      fp enclosing_radius = 0.0;
      for (std::vector<int>::const_iterator parent_hn = group.begin();
           parent_hn != group.end(); ++parent_hn) {
        const struct BH_diagnostics& d =
            state.AH_data_array[*parent_hn]->BH_diagnostics;
        const fp dx = center_x - d.origin_x;
        const fp dy = center_y - d.origin_y;
        const fp dz = center_z - d.origin_z;
        enclosing_radius =
            std::max(enclosing_radius,
                     sqrt(dx*dx + dy*dy + dz*dz) + d.max_radius);
      }
      const fp enclosing_limit =
          2.0 * merger_enclosing_radius_factor * total_mass;
      if (enclosing_radius > enclosing_limit) {
        return;
      }

      const int candidate_hn = find_unused_horizon_slot();
      if (candidate_hn == 0) {
        if (state.my_proc == 0) {
          CCTK_VWarn(CCTK_WARN_ALERT, __LINE__, __FILE__,
                     CCTK_THORNSTRING,
                     "No unused AHFinderDirect slot remains for a new "
                     "Method #2 merger candidate");
        }
        stop_search = true;
        return;
      }

      const fp candidate_radius = merger_pre_factor * total_mass;
      initialize_candidate_slot(
          CCTK_PASS_CTOC, candidate_hn, group,
          center_x, center_y, center_z, candidate_radius,
          candidate_discovery_method__method2);
      state.AH_data_array[candidate_hn]->search_flag = true;
      ++active_candidates;

      if (state.my_proc == 0) {
        CCTK_VInfo(CCTK_THORNSTRING,
                   "Created Method #2 merger candidate horizon %d with %d "
                   "parents, proxy mass %g, enclosing radius %g, and "
                   "initial radius %g",
                   candidate_hn, int(group.size()), double(total_mass),
                   double(enclosing_radius), double(candidate_radius));
      }
      stop_search =
          active_candidates >= max_active_merger_candidates;
    };
    enumerate(0, group_size);
  }
}
}

//******************************************************************************

namespace {
void release_merger_candidates(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;

  std::vector<int> candidates_to_reset;
  for (int candidate_hn = 1; candidate_hn <= N_horizons; ++candidate_hn) {
    struct AH_data& candidate = *state.AH_data_array[candidate_hn];
    if (candidate.status != horizon_status__candidate) {
      continue;
    }

    bool group_is_active = candidate.parent_horizons.size() >= 2;
    for (std::vector<int>::const_iterator parent_hn =
             candidate.parent_horizons.begin();
         group_is_active && parent_hn != candidate.parent_horizons.end();
         ++parent_hn) {
      const struct AH_data& parent = *state.AH_data_array[*parent_hn];
      group_is_active =
          parent.has_been_found &&
          !parent.inside_confirmed_merger &&
          (parent.status == horizon_status__individual ||
           parent.status == horizon_status__confirmed);
    }

    if (group_is_active &&
        candidate.candidate_method ==
            candidate_discovery_method__method1) {
      // A Method #1 group remains active while its parents are connected
      // using the enlarged release distance.  For a pair this is one edge.
      const int nparents = int(candidate.parent_horizons.size());
      std::vector<bool> reached(nparents, false);
      std::vector<int> queue(1, 0);
      reached[0] = true;
      for (std::vector<int>::size_type next = 0; next < queue.size(); ++next) {
        const int i = queue[next];
        const struct AH_data& AH_i =
            *state.AH_data_array[candidate.parent_horizons[i]];
        for (int j = 0; j < nparents; ++j) {
          if (reached[j]) {
            continue;
          }
          const struct AH_data& AH_j =
              *state.AH_data_array[candidate.parent_horizons[j]];
          const fp dx = AH_i.BH_diagnostics.centroid_x -
                        AH_j.BH_diagnostics.centroid_x;
          const fp dy = AH_i.BH_diagnostics.centroid_y -
                        AH_j.BH_diagnostics.centroid_y;
          const fp dz = AH_i.BH_diagnostics.centroid_z -
                        AH_j.BH_diagnostics.centroid_z;
          const fp distance = sqrt(dx*dx + dy*dy + dz*dz);
          const fp release_distance =
              4.0 * merger_release_factor * merger_search_factor *
              (AH_i.mass + AH_j.mass);
          if (distance < release_distance) {
            reached[j] = true;
            queue.push_back(j);
          }
        }
      }
      for (int i = 0; i < nparents; ++i) {
        group_is_active = group_is_active && reached[i];
      }
    } else if (group_is_active &&
               candidate.candidate_method ==
                   candidate_discovery_method__method2) {
      fp total_mass = 0.0;
      for (std::vector<int>::const_iterator parent_hn =
               candidate.parent_horizons.begin();
           parent_hn != candidate.parent_horizons.end(); ++parent_hn) {
        total_mass += state.AH_data_array[*parent_hn]->mass;
      }

      const fp pair_limit =
          4.0 * merger_release_factor *
          merger_compact_search_factor * total_mass;
      for (std::vector<int>::size_type i = 0;
           group_is_active && i < candidate.parent_horizons.size(); ++i) {
        const struct BH_diagnostics& di =
            state.AH_data_array[candidate.parent_horizons[i]]->BH_diagnostics;
        for (std::vector<int>::size_type j = i + 1;
             group_is_active && j < candidate.parent_horizons.size(); ++j) {
          const struct BH_diagnostics& dj =
              state.AH_data_array[candidate.parent_horizons[j]]->BH_diagnostics;
          const fp dx = di.centroid_x - dj.centroid_x;
          const fp dy = di.centroid_y - dj.centroid_y;
          const fp dz = di.centroid_z - dj.centroid_z;
          group_is_active =
              sqrt(dx*dx + dy*dy + dz*dz) <= pair_limit;
        }
      }

      if (group_is_active) {
        const struct BH_diagnostics& first =
            state.AH_data_array[candidate.parent_horizons[0]]->BH_diagnostics;
        fp min_x = first.origin_x - first.max_radius;
        fp max_x = first.origin_x + first.max_radius;
        fp min_y = first.origin_y - first.max_radius;
        fp max_y = first.origin_y + first.max_radius;
        fp min_z = first.origin_z - first.max_radius;
        fp max_z = first.origin_z + first.max_radius;
        for (std::vector<int>::size_type i = 1;
             i < candidate.parent_horizons.size(); ++i) {
          const struct BH_diagnostics& d =
              state.AH_data_array[candidate.parent_horizons[i]]
                  ->BH_diagnostics;
          min_x = std::min(min_x, d.origin_x - d.max_radius);
          max_x = std::max(max_x, d.origin_x + d.max_radius);
          min_y = std::min(min_y, d.origin_y - d.max_radius);
          max_y = std::max(max_y, d.origin_y + d.max_radius);
          min_z = std::min(min_z, d.origin_z - d.max_radius);
          max_z = std::max(max_z, d.origin_z + d.max_radius);
        }
        const fp center_x = 0.5 * (min_x + max_x);
        const fp center_y = 0.5 * (min_y + max_y);
        const fp center_z = 0.5 * (min_z + max_z);
        fp enclosing_radius = 0.0;
        for (std::vector<int>::const_iterator parent_hn =
                 candidate.parent_horizons.begin();
             parent_hn != candidate.parent_horizons.end(); ++parent_hn) {
          const struct BH_diagnostics& d =
              state.AH_data_array[*parent_hn]->BH_diagnostics;
          const fp dx = center_x - d.origin_x;
          const fp dy = center_y - d.origin_y;
          const fp dz = center_z - d.origin_z;
          enclosing_radius =
              std::max(enclosing_radius,
                       sqrt(dx*dx + dy*dy + dz*dz) + d.max_radius);
        }
        group_is_active =
            enclosing_radius <=
            2.0 * merger_release_factor *
            merger_enclosing_radius_factor * total_mass;
      }
    } else if (group_is_active) {
      CCTK_VWarn(FATAL_ERROR, __LINE__, __FILE__, CCTK_THORNSTRING,
                 "Candidate horizon %d has invalid discovery method %d",
                 candidate_hn, int(candidate.candidate_method));
    }

    if (group_is_active) {
      candidate.candidate_inactive_checks = 0;
    } else {
      ++candidate.candidate_inactive_checks;
      if (candidate.candidate_inactive_checks >= merger_release_checks) {
        candidates_to_reset.push_back(candidate_hn);
      }
    }
  }

  for (std::vector<int>::const_iterator candidate_hn =
           candidates_to_reset.begin();
       candidate_hn != candidates_to_reset.end(); ++candidate_hn) {
    if (state.my_proc == 0) {
      CCTK_VInfo(CCTK_THORNSTRING,
                 "Releasing unconverged candidate horizon %d after %d "
                 "consecutive inactive checks",
                 *candidate_hn, int(merger_release_checks));
    }
    reset_candidate_slot(CCTK_PASS_CTOC, *candidate_hn);
  }
}
}

//******************************************************************************

//
// This function is called by the Cactus scheduler to import
// the excision mask.
//
//TODO: Modernize old mask infrastructure
extern "C"
  void AHFinderDirect_import_mask(CCTK_ARGUMENTS)
{
using namespace Loop;
using namespace std;

DECLARE_CCTK_ARGUMENTS_AHFinderDirect_import_mask;
DECLARE_CCTK_PARAMETERS;

//assert(ahmask != 0);

const array<int, dim> indextype = {0, 0, 0};
const GF3D2layout layout(cctkGH, indextype);

const GF3D2<CCTK_REAL> ahmask_(layout, ahmask);
const GridDescBaseDevice grid(cctkGH);
grid.loop_all_device<0, 0, 0>(grid.nghostzones,
                                [=] CCTK_DEVICE(const PointDesc &p)
                                    CCTK_ATTRIBUTE_ALWAYS_INLINE {
																			ahmask_(p.I) = 0;
                                    });

// for (int k=0; k<cctk_lsh[2]; ++k)
// for (int j=0; j<cctk_lsh[1]; ++j)
// for (int i=0; i<cctk_lsh[0]; ++i)
// {
//         const int ind = CCTK_GFINDEX3D(cctkGH,i,j,k);
//         // zero means: point can be used,
//         // non-zero means: point must be avoided
//         ahmask[ind] = 0;
//         // if (use_mask)
//            // grid points with mask values of 1.0 can be used,
//            // values of 0.0 and 0.5 must be avoided.
//            // the excision boundary cannot be used because
//            // (a) it is inaccurate
//            // (b) it does not respect the symmetries e.g. in Kerr.
//            // then ahmask[ind] = fabs(emask[ind] - 1.0) > 0.01;
// }
}

//******************************************************************************

//
// This function is called by the Cactus scheduler to find the apparent
// horizon or horizons in the current slice.
//
extern "C"
  void AHFinderDirect_find_horizons(CCTK_ARGUMENTS)
{
DECLARE_CCTK_ARGUMENTS_AHFinderDirect_find_horizons
DECLARE_CCTK_PARAMETERS

// determine whether a horizon should be found at this iteration
bool find_any = false;
for (int hn = 1; hn <= state.my_hs->N_horizons(); ++ hn)
{
  // only try to find horizons every  find_every  time steps
  const int my_find_after = find_after_individual[hn];
  const int my_dont_find_after = dont_find_after_individual[hn];
  const fp my_find_after_time = find_after_individual_time[hn];
  const fp my_dont_find_after_time = dont_find_after_individual_time[hn];
  const int my_find_every = (find_every_individual[hn] >= 0
                             ? find_every_individual[hn]
                             : find_every);
  struct AH_data& AH_data = *state.AH_data_array[hn];
  const bool lifecycle_allows_search =
                            AH_data.status == horizon_status__individual
                         || AH_data.status == horizon_status__candidate
                         || AH_data.status == horizon_status__confirmed;
  const bool find_this =    lifecycle_allows_search
                         && cctk_iteration >= my_find_after
                         && (my_dont_find_after < 0
                             ? true
                             : cctk_iteration <= my_dont_find_after)
                         && cctk_time >= my_find_after_time
                         && (my_dont_find_after_time <= my_find_after_time
                             ? true
                             : cctk_time <= my_dont_find_after_time)
                         && my_find_every > 0
                         && cctk_iteration % my_find_every == 0
                         && (AH_data.status != horizon_status__individual
                             || ! disable_horizon[hn]);
  AH_data.search_flag = find_this;
  find_any = find_any || find_this;
}
if (! find_any) return;

if (state.timer_handle >= 0)
   then CCTK_TimerResetI(state.timer_handle);

const int my_proc = state.my_proc;
horizon_sequence& hs = *state.my_hs;
const bool broadcast_horizon_shape = true;
bool dynamic_horizon_assignment = state.dynamic_horizon_assignment;
if (dynamic_horizon_assignment)
   then {
	for (int hn = 1 ; hn <= N_horizons ; ++hn)
	  if (state.AH_data_array[hn]->search_flag
	      && state.AH_data_array[hn]->use_pretracking)
	     then {
		  dynamic_horizon_assignment = false;
		  if (my_proc == 0 && cctk_iteration == 0)
		     then CCTK_VWarn(1, __LINE__, __FILE__, CCTK_THORNSTRING,
			     "dynamic horizon assignment does not support "
			     "pretracking; using static assignment");
		  break;
		  }
	}
const bool active_flag = dynamic_horizon_assignment
			 ? my_proc < state.N_active_procs
			 : hs.has_genuine_horizons();

      struct cactus_grid_info&          cgi = state.cgi;
const struct    geometry_info&           gi = state.gi;
      struct    Jacobian_info&     Jac_info = state.Jac_info;
      struct          IO_info&      IO_info = state.IO_info;
const struct       error_info&   error_info = state.error_info;
const struct     verbose_info& verbose_info = state.verbose_info;

// what are the semantics of the Cactus gxx variables? (these may
// change from one call to another, so we have to re-check each time)

#if 0
if      (CCTK_Equals(metric_type, "physical"))
   then cgi.use_Cactus_conformal_metric = false;
else if (CCTK_Equals(metric_type, "static conformal"))
   then cgi.use_Cactus_conformal_metric = (*conformal_state > 0);
else	CCTK_VWarn(FATAL_ERROR, __LINE__, __FILE__, CCTK_THORNSTRING,
"AHFinderDirect_find_horizons(): unknown metric_type=\"%s\"!",
		   metric_type);				/*NOTREACHED*/
#endif
cgi.use_Cactus_conformal_metric = false; // CarpetX Change: disable multiple metric types

// update parameters
IO_info.output_ASCII_files = (output_ASCII_files != 0);
IO_info.output_HDF5_files = (output_HDF5_files != 0);
IO_info.output_initial_guess = (output_initial_guess != 0);
IO_info.output_h_every     = output_h_every;
IO_info.output_Theta_every = output_Theta_every;
IO_info.output_mean_curvature_every = output_mean_curvature_every;
IO_info.output_h     = false;	// dummy value
IO_info.output_Theta = false;	// dummy value
IO_info.output_mean_curvature = false;	// dummy value

IO_info.output_BH_diagnostics              = (output_BH_diagnostics != 0);
IO_info.BH_diagnostics_directory
	= (strlen(BH_diagnostics_directory) == 0)
	  ? /* IO:: */ out_dir
	  : BH_diagnostics_directory;
IO_info.BH_diagnostics_base_file_name      = BH_diagnostics_base_file_name;
IO_info.BH_diagnostics_file_name_extension = BH_diagnostics_file_name_extension;

IO_info.output_ghost_zones_for_h  = (output_ghost_zones_for_h != 0);
IO_info.ASCII_gnuplot_file_name_extension = ASCII_gnuplot_file_name_extension;
IO_info.HDF5_file_name_extension          = HDF5_file_name_extension;
IO_info.h_directory
	= (strlen(h_directory) == 0)
	  ? /* IO:: */ out_dir
	  : h_directory;
IO_info.h_base_file_name         = h_base_file_name;
IO_info.Theta_base_file_name     = Theta_base_file_name;
IO_info.mean_curvature_base_file_name     = mean_curvature_base_file_name;
IO_info.Delta_h_base_file_name   = Delta_h_base_file_name;
IO_info.h_min_digits             = h_min_digits;
IO_info.Jacobian_base_file_name  = Jacobian_base_file_name;
IO_info.output_OpenDX_control_files  = (output_OpenDX_control_files != 0);
IO_info.OpenDX_control_file_name_extension = OpenDX_control_file_name_extension;
IO_info.time_iteration = 0;
IO_info.time           = 0.0;

// get the Cactus time step and decide if we want to output h and/or Theta now
IO_info.time_iteration = cctk_iteration;
IO_info.time           = cctk_time;
IO_info.output_h
   = (IO_info.output_h_every > 0)
     && ((IO_info.time_iteration % IO_info.output_h_every) == 0);
IO_info.output_Theta
   = (IO_info.output_Theta_every > 0)
     && ((IO_info.time_iteration % IO_info.output_Theta_every) == 0);
IO_info.output_mean_curvature
   = (IO_info.output_mean_curvature_every > 0)
     && ((IO_info.time_iteration % IO_info.output_mean_curvature_every) == 0);

// Discover merger candidates before preparing initial guesses and running
// Newton.  Newly created candidates have search_flag set by the discovery
// routines, so they are solved in this same horizon-finder call.
if (state.method == method__find_horizons)
   then {
        if (cctk_iteration >= merger_discovery_min_iteration) {
	  discover_method1_mergers(CCTK_PASS_CTOC);
	  discover_method2_mergers(CCTK_PASS_CTOC);
        } else if (discover_mergers && state.my_proc == 0) {
          CCTK_VInfo(CCTK_THORNSTRING,
                     "Skipping dynamical merger discovery at iteration %d; "
                     "merger_discovery_min_iteration=%d",
                     int(cctk_iteration),
                     int(merger_discovery_min_iteration));
        }
	}
	if (AH_set_origins_to_punctures) {
	  set_search_origins_from_punctures(CCTK_PASS_CTOC);
	}

// set initial guess for any (genuine) horizons that need it,
// i.e. for any (genuine) horizons where we didn't find the horizon previously
	for (int hn = hs.init_hn() ; hs.is_genuine() ; hn = hs.next_hn())
	{
	if (!dynamic_horizon_assignment && !hs.is_hn_genuine(hn))
	   then continue;
	assert( state.AH_data_array[hn] != NULL );
	struct AH_data& AH_data = *state.AH_data_array[hn];
	if (!AH_data.search_flag)
	   then continue;
        if (verbose_info.print_algorithm_details) {
          printf ("AHF find_horizons[%d] initial_find_flag=%d\n", hn, (int) AH_data.initial_find_flag);
          printf ("AHF find_horizons[%d] really_initial_find_flag=%d\n", hn, (int) AH_data.really_initial_find_flag);
          printf ("AHF find_horizons[%d] search_flag=%d\n", hn, (int) AH_data.search_flag);
          printf ("AHF find_horizons[%d] found_flag=%d\n", hn, (int) AH_data.found_flag);
        }
	if (AH_data.found_flag)
           then {
                AH_data.initial_find_flag = false;
                AH_data.really_initial_find_flag = false;
                }
	   else {
                if (AH_data.really_initial_find_flag
                    || AH_data.initial_guess_info.reset_horizon_after_not_finding)
                   then {
		        patch_system& ps = *AH_data.ps_ptr;
                        if (verbose_info.print_algorithm_details) {
                          printf ("AHF find_horizons[%d] setup_initial_guess\n", hn);
                        }
                        if (track_origin_from_grid_scalar[hn] &&
                            !AH_set_origins_to_punctures &&
                            state.method == method__find_horizons) {
                           track_origin(cctkGH, ps, &AH_data, hn, verbose_info.print_algorithm_details);
                           set_initial_guess_parameters(AH_data, hn, 
                                                        ps.origin_x(), ps.origin_y(), ps.origin_z());
                        }
        		setup_initial_guess(ps,
        		          	    AH_data.initial_guess_info,
        				    IO_info,
        				    hn, N_horizons, verbose_info);
                if (cctk_iteration == 0 && my_proc == 0) {
                  jtutil::norm<fp> h_norms;
                  ps.ghosted_gridfn_norms(gfns::gfn__h, h_norms);
                  CCTK_VInfo(
                      CCTK_THORNSTRING,
                      "Initial AH interpolation surface %d: "
                      "origin=(%.17g,%.17g,%.17g), "
                      "radius range=[%.17g,%.17g]",
                      hn, double(ps.origin_x()), double(ps.origin_y()),
                      double(ps.origin_z()), double(h_norms.min_abs_value()),
                      double(h_norms.max_abs_value()));
                }
		if (active_flag && IO_info.output_initial_guess
			    && (!dynamic_horizon_assignment || my_proc == 0))
        		   then output_gridfn(ps, gfns::gfn__h,
                                              "h", cgi.GH,
        				      IO_info, IO_info.h_base_file_name,
                                              IO_info.h_min_digits,
        				      hn, verbose_info
        				      .print_algorithm_highlights);
        		AH_data.initial_find_flag = true;
        		}
                }
	}

//
// now the main horizon finding (or other computation)
//
switch	(state.method)
	{
case method__evaluate_expansions:
	do_evaluate_expansions(my_proc, N_horizons,
			       *state.my_hs, state.AH_data_array,
			       cgi, gi, IO_info,
			       error_info, verbose_info,
			       state.timer_handle);
	break;

case method__test_expansion_Jacobians:
	do_test_expansion_Jacobians(my_proc, N_horizons,
				    state.AH_data_array,
				    cgi, gi, Jac_info,
				    (test_all_Jacobian_compute_methods != 0),
				    IO_info, error_info, verbose_info,
				    state.timer_handle);
	break;

case method__find_horizons:
	  {
	if (state.timer_handle >= 0)
	   then CCTK_TimerStartI(state.timer_handle);
	Newton(cctkGH,
	       state.N_procs, state.N_active_procs, my_proc,
	       dynamic_horizon_assignment,
	       *state.my_hs, state.AH_data_array,
	       cgi, gi, Jac_info, state.solver_info,
	       IO_info, state.BH_diagnostics_info, broadcast_horizon_shape,
	       error_info, verbose_info,
	       state.isb);
	std::vector<int> converged_candidates;
	for (int hn = 1; hn <= N_horizons; ++hn)
	  {
	  struct AH_data& AH_data = *state.AH_data_array[hn];
	  if (AH_data.found_flag)
	     then {
		  if (AH_data.status == horizon_status__candidate)
		     then converged_candidates.push_back(hn);
		  else AH_data.has_been_found = true;
		  }
	  else if (AH_data.search_flag &&
	           AH_data.status == horizon_status__candidate)
	     then ++AH_data.candidate_failed_searches;
	  }
	std::sort(converged_candidates.begin(), converged_candidates.end(),
	          [](const int a, const int b) {
	            const std::size_t size_a =
	                state.AH_data_array[a]->parent_horizons.size();
	            const std::size_t size_b =
	                state.AH_data_array[b]->parent_horizons.size();
	            return size_a != size_b ? size_a < size_b : a < b;
	          });
	for (std::vector<int>::const_iterator candidate_hn =
	         converged_candidates.begin();
	     candidate_hn != converged_candidates.end(); ++candidate_hn)
	  {
	  const struct AH_data& candidate = *state.AH_data_array[*candidate_hn];
	  bool parent_was_enclosed = false;
	  for (std::vector<int>::const_iterator parent_hn =
	           candidate.parent_horizons.begin();
	       parent_hn != candidate.parent_horizons.end(); ++parent_hn)
	    {
	    parent_was_enclosed =
	        parent_was_enclosed ||
	        state.AH_data_array[*parent_hn]->inside_confirmed_merger;
	    }
	  if (parent_was_enclosed)
	     then {
		  if (state.my_proc == 0)
		     then CCTK_VInfo(
		         CCTK_THORNSTRING,
		         "Discarding simultaneously converged candidate horizon %d "
		         "because a smaller overlapping candidate was promoted first",
		         *candidate_hn);
		  reset_candidate_slot(CCTK_PASS_CTOC, *candidate_hn);
		  }
	  else {
		promote_candidate_slot(*candidate_hn);
		write_merger_event(CCTK_PASS_CTOC, *candidate_hn);
		}
	  }
	// Release is deliberately last: every candidate discovered above first
	// receives a Newton solve and a chance to be promoted on this iteration.
	release_merger_candidates(CCTK_PASS_CTOC);
	if (state.timer_handle >= 0)
	   then CCTK_TimerStopI(state.timer_handle);
	break;
	  }

default:
	CCTK_VWarn(FATAL_ERROR, __LINE__, __FILE__, CCTK_THORNSTRING,
"\n"
"   find_horizons(): unknown method=(int)%d!\n"
"                    (this should never happen!)"
		   ,
		   int(state.method));				/*NOTREACHED*/
	}

if (state.timer_handle >= 0)
   then {
	CCTK_VInfo(CCTK_THORNSTRING,
		   "timer stats for computation:");
	CCTK_TimerPrintDataI(state.timer_handle, -1);
	}
}

//******************************************************************************
//******************************************************************************
//******************************************************************************

//
// This function implements  AHFinderDirect::method == "horizon function":
// On processor #0 it evaluates the Theta(h) function for each apparent
// horizon (and does any I/O desired); on other processors it does N_horizons
// dummy evaluations on horizon #0.
//
// Note that if we decide to output h, we output it *after* any Theta(h)
// evaluation or horizon finding has been done, to ensure that all the
// ghost zones are filled in in case we need to print them.
//
// Arguments:
// timer_handle = a valid Cactus timer handle if we want to time the
//		  apparent horizon process, or -ve to skip this
//		  (we only time the computation, not the file I/O)
//
namespace {
void do_evaluate_expansions(int my_proc, int N_horizons,
			    horizon_sequence& hs,
			       struct AH_data* const AH_data_array[],
			    const struct cactus_grid_info& cgi,
			    const struct geometry_info& gi,
			    const struct IO_info& IO_info,
			    const struct error_info& error_info,
			    const struct verbose_info& verbose_info,
			    int timer_handle)
{
const bool active_flag = (my_proc == 0);

if (active_flag)
   then {
	assert( hs.N_horizons() == N_horizons );
	assert( hs.my_N_horizons() == N_horizons );

		for (int hn = hs.init_hn() ;
		     hs.is_genuine() ;
		     hn = hs.next_hn())
		{
		assert( AH_data_array[hn] != NULL );
		struct AH_data& AH_data = *AH_data_array[hn];
		patch_system& ps = *AH_data.ps_ptr;

		if (timer_handle >= 0)
		   then CCTK_TimerStartI(timer_handle);
		jtutil::norm<fp> Theta_norms;
		const bool Theta_ok = expansion(&ps,
                                                AH_data.compute_info,
						cgi, gi,
						error_info, true,// initial eval
						false,	// no Jacobian coeffs
						true,	// yes, print msgs
						&Theta_norms);
		if (timer_handle >= 0)
		   then CCTK_TimerStopI(timer_handle);

		if (IO_info.output_h)
		   then output_gridfn(ps, gfns::gfn__h,
                                      "h", cgi.GH,
				      IO_info, IO_info.h_base_file_name,
                                      IO_info.h_min_digits,
				      hn, verbose_info.print_algorithm_details);

		if (Theta_ok)
		   then {
			CCTK_VInfo(CCTK_THORNSTRING,
			   "   Theta(h) rms-norm %.2e, infinity-norm %.2e",
			   Theta_norms.rms_norm(), Theta_norms.infinity_norm());
			if (IO_info.output_Theta)
			   then output_gridfn(ps, gfns::gfn__Theta,
                                              "Theta", cgi.GH,
					      IO_info, IO_info
						       .Theta_base_file_name,
                                              IO_info.h_min_digits,
					      hn, verbose_info
						  .print_algorithm_details);
			if (IO_info.output_mean_curvature)
			   then output_gridfn(ps, gfns::gfn__mean_curvature,
                                              "mean_curvature", cgi.GH,
					      IO_info, IO_info
						       .mean_curvature_base_file_name,
                                              IO_info.h_min_digits,
					      hn, verbose_info
						  .print_algorithm_details);
			}
		}
	}
   else {
                struct what_to_compute new_compute_info;
		for (int i = 0 ; i < N_horizons ; ++i)
		{
                expansion(NULL, new_compute_info,
			  cgi, gi,
			  error_info, true);	// initial evaluation
		}
	}
}
	  }

//******************************************************************************

//
// This function implements
//  AHFinderDirect::method == "test expansion Jacobians":
// On processor #0 it computes and prints the Jacobian matrix J[Theta(h)]
// function for horizon #1; on other processors it does dummy Jacobian
// computations.
//
// The Jacobian computation may optionally be done in several different
// ways, in which case all the resulting Jacobian matrices are printed,
// as are their differences.  Alternatively, only
// the numerical perturbation computation may be done/printed.
//
// Arguments:
// timer_handle = a valid Cactus timer handle if we want to time the
//		  apparent horizon process, or -ve to skip this
//		  (we only time the computation, not the file I/O)
// test_all_Jacobian_compute_methods
//	= true ==> Test all known methods of computing the Jacobian
//		   matrix, and print all the resulting Jacobian matrices
//		   and their differences.
//	  false ==> Just test/print the numerical perturbation calculation.
//		    (This may be useful if one or more of the other methods
//		    is broken.)
//
namespace {
void do_test_expansion_Jacobians(int my_proc, int N_horizons,
				 struct AH_data* const AH_data_array[],
				 const struct cactus_grid_info& cgi,
				 const struct geometry_info& gi,
				       struct Jacobian_info& Jac_info,
				 bool test_all_Jacobian_compute_methods,
				 const struct IO_info& IO_info,
				 const struct error_info& error_info,
				 const struct verbose_info& verbose_info,
				 int timer_handle)
{
const bool active_flag = (my_proc == 0);
assert(N_horizons >= 1);

const bool print_msg_flag = true;
const int hn = 1;

struct AH_data* const AH_data_ptr = active_flag ? AH_data_array[hn]   : NULL;
patch_system*   const      ps_ptr = active_flag ? AH_data_ptr->ps_ptr : NULL;
struct what_to_compute dummy_compute_info;
struct what_to_compute & compute_info =
	active_flag
	? AH_data_ptr->compute_info
	: dummy_compute_info;

//
// numerical-perturbation Jacobian
//
Jacobian* Jac_NP_ptr = active_flag ? AH_data_ptr->Jac_ptr : NULL;
expansion(ps_ptr, compute_info,
	  cgi, gi,
	  error_info, true);		// initial evaluation
Jac_info.Jacobian_compute_method = Jacobian__numerical_perturbation;
expansion_Jacobian(ps_ptr, Jac_NP_ptr,
                   compute_info,
		   cgi, gi, Jac_info,
		   error_info, true,	// initial evaluation
		   print_msg_flag);

Jacobian* Jac_SD_FDdr_ptr = NULL;
if (test_all_Jacobian_compute_methods)
   then {
	// symbolic differentiation with finite diff d/dr
	Jac_SD_FDdr_ptr = active_flag
			  ? new_Jacobian(Jac_info.Jacobian_store_solve_method,
					 *ps_ptr,
					 verbose_info.print_algorithm_details)
			  : NULL;
	expansion(ps_ptr, compute_info,
		  cgi, gi,
		  error_info, true,	// initial evaluation
		  true);		// compute SD Jacobian coeffs
	Jac_info.Jacobian_compute_method = Jacobian__symbolic_diff_with_FD_dr;
	expansion_Jacobian(ps_ptr, Jac_SD_FDdr_ptr,
                           compute_info,
			   cgi, gi, Jac_info,
			   error_info, true,	// initial evaluation
			   print_msg_flag);
	}

if (active_flag)
   then output_Jacobians(*ps_ptr,
			 Jac_NP_ptr, Jac_SD_FDdr_ptr,
			 IO_info, IO_info.Jacobian_base_file_name,
			 IO_info.h_min_digits,
			 hn, print_msg_flag);
}
	  }

//******************************************************************************

	  }	// namespace AHFinderDirect
