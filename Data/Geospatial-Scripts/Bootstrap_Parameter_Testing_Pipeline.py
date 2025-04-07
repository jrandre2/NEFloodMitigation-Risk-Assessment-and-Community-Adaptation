# --- Imports ---
import arcpy
import numpy as np
import os
import datetime
import time
import pprint
from collections import defaultdict # Useful for aggregating counts

print("Starting Optimized Bootstrap Pipeline Script...")

# --- Configuration Dictionary ---
# ... (config remains the same) ...

# --- Helper Functions ---
# ... (validate_inputs, load_prepare_buildings, load_prepare_claims,
#      find_matching_buildings, run_policy_bootstrap, calculate_stats,
#      calculate_accuracy_metrics remain mostly the same, but check if
#      load_prepare_buildings needs adjustment based on output strategy) ...

# --- OPTIMIZED Output Function (Example: Table Output - Strategy 2B) ---
def write_output_table(config, all_iteration_counts, buildings_dict):
    """Creates a standalone table with bootstrap counts for all iterations."""
    print("Writing output table with counts for all iterations...")
    out_gdb = config["outputs"]["output_gdb"]
    # Example table name - adjust as needed
    base_name = config['base_run_name']
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    out_table_name = arcpy.ValidateTableName(f"BootstrapCounts_{base_name}_{ts}", out_gdb)
    out_table_path = os.path.join(out_gdb, out_table_name)
    bldg_id_field = config["buildings"]["id_field"]
    bldg_id_field_type = "TEXT" # Assume text, adjust if numeric
    # Check BldgID field type from original data
    try:
        desc = arcpy.da.Describe(config["buildings"]["path"])
        for field in desc['fields']:
             if field.name.lower() == bldg_id_field.lower():
                 if field.type in ['Integer', 'SmallInteger', 'OID']:
                     bldg_id_field_type = "LONG"
                 elif field.type in ['Double', 'Single']:
                     bldg_id_field_type = "DOUBLE"
                 # Add other types if necessary
                 break
    except Exception:
        print(f"Warning: Could not determine exact type for {bldg_id_field}, defaulting to TEXT.")


    if arcpy.Exists(out_table_path):
        print(f"Deleting existing output table: {out_table_path}")
        arcpy.management.Delete(out_table_path)

    print(f"Creating output table: {out_table_path}")
    arcpy.management.CreateTable(out_gdb, out_table_name)

    # Add Building ID field
    arcpy.management.AddField(out_table_path, bldg_id_field, bldg_id_field_type, field_alias=bldg_id_field)

    # Add count fields for each iteration tested
    iteration_fields = []
    for n_iter in config["n_iterations_to_test"]:
        count_field_name = f"Count_{n_iter}Iter"
        # Ensure field name is valid (ArcGIS limits length and chars)
        valid_count_field_name = arcpy.ValidateFieldName(count_field_name, out_gdb)
        if valid_count_field_name != count_field_name:
            print(f"Warning: Adjusted count field name '{count_field_name}' to '{valid_count_field_name}'")
        arcpy.management.AddField(out_table_path, valid_count_field_name, "LONG", field_alias=f"Bootstrap Count ({n_iter} Iter)")
        iteration_fields.append(valid_count_field_name) # Store the valid name used

    # --- Efficiently populate the table ---
    # Get all unique building IDs that received counts across all iterations
    all_counted_bldg_ids = set()
    for iter_counts in all_iteration_counts.values():
        all_counted_bldg_ids.update(iter_counts.keys())

    print(f"Inserting counts for {len(all_counted_bldg_ids)} buildings into {out_table_name}...")
    insert_fields = [bldg_id_field] + iteration_fields

    insert_count = 0
    with arcpy.da.InsertCursor(out_table_path, insert_fields) as i_cursor:
        for bldg_id in all_counted_bldg_ids:
            # Create the row structure
            new_row = [None] * len(insert_fields)
            new_row[0] = bldg_id # Set the building ID

            # Populate counts for each iteration
            for i, n_iter in enumerate(config["n_iterations_to_test"]):
                # Get the count for this bldg_id and this iteration, default to 0
                count = all_iteration_counts.get(n_iter, {}).get(bldg_id, 0)
                new_row[i + 1] = count # +1 because index 0 is bldg_id

            try:
                i_cursor.insertRow(new_row)
                insert_count += 1
            except Exception as insert_err:
                print(f"Warning: Failed to insert row for BldgID {bldg_id}. Error: {insert_err}")

    print(f"Inserted {insert_count} rows into {out_table_name}.")
    if insert_count != len(all_counted_bldg_ids):
        print(f"Warning: Mismatch between expected buildings ({len(all_counted_bldg_ids)}) and inserted rows ({insert_count}).")

    return out_table_path


# --- Main Pipeline Execution ---
if __name__ == "__main__":
    overall_start_time = time.time()
    final_report_path = None
    analysis_layer = None
    residential_layer_name = "residential_buildings_lyr_temp"
    created_buffers = [] # Keep track of temporary buffers to delete
    inundated_oids_by_distance = {} # Store results of spatial selections

    try:
        # 1. Validate Inputs and Set Environment
        validate_inputs(config)
        arcpy.env.overwriteOutput = True
        # ... (rest of report path logic) ...
        # --- Setup final report path determination (as before) ---
        output_gdb_path = config["outputs"]["output_gdb"]
        report_dir = os.path.dirname(output_gdb_path) if '.' in os.path.basename(output_gdb_path) and os.path.isfile(output_gdb_path) else output_gdb_path
        if not arcpy.Exists(report_dir):
            report_dir = os.path.dirname(report_dir)
        if not arcpy.Exists(report_dir):
            report_dir = os.getcwd()
            print(f"Warning: Could not determine valid directory from output GDB path, saving report to current directory: {report_dir}")
        final_report_path = os.path.join(report_dir, config["outputs"]["final_report_name"])
        # --- End Report Path Setup ---

        # 2. Load Data (once)
        buildings_data, buildings_grouped, bldg_load_stats = load_prepare_buildings(config)
        claims_data, claims_load_stats = load_prepare_claims(config)

        # 3. Prepare Filtered Residential Layer (once)
        # ... (MakeFeatureLayer logic as before to create analysis_layer) ...
        # --- Create analysis_layer (as before) ---
        zone_field=config["buildings"]["zone_field"]
        zone_value=config["buildings"]["filter_zone_value"]
        where_clause = None
        if zone_value is not None:
            try:
                zone_field_obj = arcpy.ListFields(config["buildings"]["path"], zone_field)[0]
                delim_zone_fld = arcpy.AddFieldDelimiters(config["buildings"]["path"], zone_field)
                if zone_field_obj.type in ['String', 'GUID', 'Date']:
                    where_clause = f"{delim_zone_fld} = '{zone_value}'"
                else:
                    where_clause = f"{delim_zone_fld} = {zone_value}"
                if arcpy.Exists(residential_layer_name):
                    print(f"Deleting existing temp layer: {residential_layer_name}")
                    arcpy.management.Delete(residential_layer_name)
                arcpy.management.MakeFeatureLayer(config["buildings"]["path"], residential_layer_name, where_clause)
                analysis_layer = residential_layer_name
                print(f"Created temporary layer for selections: {analysis_layer} using filter: {where_clause}")
                count_check = int(arcpy.management.GetCount(analysis_layer)[0])
                if count_check == 0:
                    print(f"Warning: Temporary residential layer '{analysis_layer}' created with 0 features matching filter '{where_clause}'. Accuracy assessment might yield no results.")
            except Exception as e:
                print(f"Warning: Could not create filtered residential layer using Zone field. Error: {e}. Accuracy assessment might run on all buildings or fail if layer is needed.")
                analysis_layer = config["buildings"]["path"]
                print(f"Proceeding with analysis_layer as: {analysis_layer}")
        else:
            analysis_layer = config["buildings"]["path"]
            print("Info: No zone filter applied; accuracy assessment will run on all loaded buildings.")
        # --- End analysis_layer Creation ---


        # 4. ***OPTIMIZED*** Prepare Accuracy Assessment Inputs (Buffers & Selections)
        print("\nPre-calculating buffer intersections...")
        inundation_layer = config["accuracy"]["inundation_layer_name"]
        buffer_distances = config["accuracy"]["buffer_distances"]
        buffer_units = config["accuracy"]["buffer_units"]
        buffer_base_name = f"temp_acc_buffer" # Base name for temp buffers

        if not arcpy.Exists(analysis_layer):
             raise ValueError(f"Analysis layer '{analysis_layer}' does not exist before buffering.")

        total_buildings_in_analysis_layer = int(arcpy.management.GetCount(analysis_layer)[0])
        print(f"Total buildings in analysis layer for accuracy check: {total_buildings_in_analysis_layer}")

        if total_buildings_in_analysis_layer == 0:
            print("Warning: Analysis layer for accuracy has 0 features. Skipping accuracy pre-calculation.")
        else:
            for distance in buffer_distances:
                buffer_start_time = time.time()
                print(f"  Processing Buffer Distance: {distance} {buffer_units}...")
                buffer_poly_for_select = None
                buffer_output_path = None # Define path variable

                try:
                    if distance == 0:
                        buffer_poly_for_select = inundation_layer
                        print(f"    Using original inundation layer for distance 0.")
                    else:
                        buffer_output_name = arcpy.ValidateTableName(f"{buffer_base_name}_{distance}".replace('.', '_'), config["outputs"]["output_gdb"])
                        buffer_output_path = os.path.join(config["outputs"]["output_gdb"], buffer_output_name)
                        buffer_distance_with_units = f"{distance} {buffer_units}".strip()

                        if arcpy.Exists(buffer_output_path):
                            arcpy.management.Delete(buffer_output_path)

                        print(f"    Creating buffer: {buffer_output_path}...")
                        arcpy.analysis.Buffer(inundation_layer, buffer_output_path, buffer_distance_with_units, dissolve_option="ALL")
                        buffer_poly_for_select = buffer_output_path
                        created_buffers.append(buffer_output_path) # Add to list for cleanup
                        print(f"    Buffer created.")

                    # Perform selection and store OIDs
                    print(f"    Selecting buildings intersecting buffer/inundation...")
                    arcpy.management.SelectLayerByAttribute(analysis_layer, "CLEAR_SELECTION") # Ensure clear selection
                    selection_result = arcpy.management.SelectLayerByLocation(analysis_layer, "INTERSECT", buffer_poly_for_select, selection_type="NEW_SELECTION")
                    intersect_count = int(arcpy.management.GetCount(selection_result)[0])
                    print(f"    Found {intersect_count} intersecting buildings.")

                    selected_oids = set()
                    if intersect_count > 0:
                        # Use OID field name from the description of analysis_layer
                        # This handles cases where it might not be 'OID' (though usually is)
                        oid_field_name = arcpy.Describe(analysis_layer).OIDFieldName
                        with arcpy.da.SearchCursor(selection_result, [oid_field_name]) as sel_cursor:
                            for sel_row in sel_cursor:
                                selected_oids.add(sel_row[0])

                    inundated_oids_by_distance[distance] = selected_oids
                    arcpy.management.SelectLayerByAttribute(analysis_layer, "CLEAR_SELECTION") # Clear selection for next buffer
                    print(f"    Stored {len(selected_oids)} OIDs for distance {distance}.")

                except Exception as precalc_err:
                    print(f"  ERROR pre-calculating for buffer {distance} {buffer_units}: {precalc_err}")
                    inundated_oids_by_distance[distance] = set() # Store empty set on error? Or re-raise?
                    # Optionally delete partially created buffer if error occurred during selection
                    if buffer_output_path and arcpy.Exists(buffer_output_path):
                         try: arcpy.management.Delete(buffer_output_path)
                         except: pass # Ignore delete error here
                    # Decide if you want to continue or stop on error
                    # raise precalc_err # Uncomment to stop execution on error

                print(f"  Buffer distance {distance} pre-calculation finished in {time.time() - buffer_start_time:.2f} sec.")

        print("Finished pre-calculating buffer intersections.\n")


        # 5. Open Final Report File
        print(f"Opening final report file for writing: {final_report_path}")
        all_results_aggregated = {} # Store counts for all iterations {n_iter: {bldg_id: count, ...}, ...}

        with open(final_report_path, 'w') as report_file:
            # ... (Write Overall Header & Config/Load Summaries - as before) ...
            report_file.write("="*70 + "\n"); report_file.write(" Bootstrap Pipeline & Accuracy Sensitivity Report (RESIDENTIAL ONLY)\n"); report_file.write("="*70 + "\n")
            report_file.write(f"Date Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            report_file.write("\n" + "-"*70 + "\n"); report_file.write("Base Configuration Used:\n")
            base_report_config = config.copy() ; base_report_config['parameters']['n_iterations'] = '*Varied in runs below*'
            report_file.write(pprint.pformat(base_report_config, indent=2, width=100)); report_file.write("\n" + "-"*70 + "\n\n")
            report_file.write("Data Loading & Filtering Summary:\n"); report_file.write(f"  Buildings Read Initially: {bldg_load_stats.get('initial_read', 'N/A')}\n")
            report_file.write(f"  Buildings Skipped (Invalid Data): {bldg_load_stats.get('skipped_invalid_data', 'N/A')}\n"); report_file.write(f"  Buildings Skipped (Zone Filter): {bldg_load_stats.get('skipped_zone_filter', 'N/A')}\n")
            report_file.write(f"  Buildings Skipped (Not Largest): {bldg_load_stats.get('skipped_not_largest', 'N/A')}\n"); report_file.write(f"  Buildings Loaded for Matching: {bldg_load_stats.get('final_processed_count', 'N/A')}\n")
            report_file.write(f"  Claims Loaded (Event Filtered): {claims_load_stats.get('processed', 'N/A')}\n"); report_file.write(f"  Claims Skipped (Data Issues): {claims_load_stats.get('skipped', 'N/A')}\n"); report_file.write("-" * 70 + "\n\n")


            # 6. --- Loop Over Iteration Counts ---
            for n_iter in config["n_iterations_to_test"]:
                iteration_start_time = time.time()
                current_run_name = f"{config['base_run_name']}_{n_iter}Iter"
                print(f"\n===== Starting Run for {n_iter} Iterations ({current_run_name}) =====")
                report_file.write("*"*15 + f" Results for N_Iterations = {n_iter} " + "*"*15 + "\n\n")

                # Store iteration-specific stats
                run_stats_iter = { 'claims_matched_attempted': 0, 'claims_with_matches': 0, 'claims_no_matches': 0, 'matches_per_policy_list': [] }

                try:
                    # --- Run Core Bootstrap Logic ---
                    print(f"  Running matching and bootstrapping ({n_iter} iterations)...")
                    overall_building_counts_iter = defaultdict(int) # Use defaultdict for easier counting
                    for i, policy in enumerate(claims_data):
                        run_stats_iter['claims_matched_attempted'] += 1
                        matches = find_matching_buildings(policy, buildings_data, buildings_grouped, {"elevation_tolerance_abs": config["parameters"]["elevation_tolerance_abs"]})
                        if matches:
                            run_stats_iter['claims_with_matches'] += 1; run_stats_iter['matches_per_policy_list'].append(len(matches))
                            policy_counts = run_policy_bootstrap(matches, n_iter)
                            for bldg_id, count in policy_counts.items():
                                overall_building_counts_iter[bldg_id] += count # Increment count
                        else:
                            run_stats_iter['claims_no_matches'] += 1
                    print("  Finished matching and bootstrapping.")

                    # Store results for this iteration for later table writing
                    all_results_aggregated[n_iter] = dict(overall_building_counts_iter) # Convert back to dict if needed

                    # --- Write Bootstrap Summary to Report ---
                    # ... (Bootstrap summary report writing as before) ...
                    report_file.write("  Bootstrap Run Summary:\n"); report_file.write(f"    Claims Processed: {run_stats_iter['claims_matched_attempted']}\n")
                    report_file.write(f"    Claims with >= 1 Match: {run_stats_iter['claims_with_matches']}\n"); report_file.write(f"    Claims with 0 Matches: {run_stats_iter['claims_no_matches']}\n")
                    matches_list = run_stats_iter['matches_per_policy_list']
                    if matches_list: match_arr = np.array(matches_list); report_file.write(f"    Min/Max/Mean/Median Matches per Policy: {np.min(match_arr)} / {np.max(match_arr)} / {np.mean(match_arr):.2f} / {np.median(match_arr)}\n")
                    else: report_file.write("    Matching Stats: No claims had matches.\n")
                    counts_list = [c for c in overall_building_counts_iter.values() if c > 0] # Get counts > 0
                    if counts_list:
                        count_arr = np.array(counts_list); report_file.write(f"    Buildings with Count > 0: {len(count_arr)}\n"); report_file.write(f"    Min/Max/Mean/Median Count (>0): {np.min(count_arr)} / {np.max(count_arr)} / {np.mean(count_arr):.2f} / {np.median(count_arr)}\n")
                        expected_sum = n_iter * run_stats_iter['claims_with_matches']; report_file.write(f"    Total Sum of Counts: {np.sum(count_arr)} (Expected: {expected_sum})\n")
                        if not np.isclose(np.sum(count_arr), expected_sum): report_file.write("    *** WARNING: Count sum mismatch! ***\n")
                    else: report_file.write("    Count Distribution: No buildings received counts > 0.\n")
                    report_file.write("-" * 60 + "\n\n")


                    # --- ***NO FC WRITING HERE*** (Using table output later) ---
                    # If you chose Solution 2A (UpdateCursor), do that here.
                    # If you chose Solution 2C (Final FC only), do nothing here.

                    # --- ***OPTIMIZED*** Run Accuracy Assessment Loop ---
                    print(f"  Calculating Accuracy Assessment vs Buffers for {n_iter} iterations...")
                    report_file.write("  Accuracy Assessment vs Buffered Inundation:\n\n")

                    # Initialize contingency tables and stats dicts for *all* buffer distances for *this* iteration
                    contingency_tables = {dist: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0} for dist in buffer_distances}
                    count_stats = {
                        dist: {'inundated_counts': [], 'non_inundated_counts': []}
                        for dist in buffer_distances
                    }
                    total_inundated_counts = {dist: 0 for dist in buffer_distances}
                    total_non_inundated_counts = {dist: 0 for dist in buffer_distances}

                    # --- Single Pass Through Buildings for Accuracy ---
                    building_count_total_for_acc = 0 # Count buildings actually checked
                    if total_buildings_in_analysis_layer > 0: # Only proceed if analysis layer isn't empty
                        for bldg_id, data in buildings_data.items():
                            oid = data.get("OID")
                            if oid is None: continue # Should not happen if loading was correct

                            building_count_total_for_acc += 1
                            count = overall_building_counts_iter.get(bldg_id, 0)
                            selected_by_model = count > 0

                            # Check against each pre-calculated buffer distance
                            for distance in buffer_distances:
                                inundated_oids = inundated_oids_by_distance.get(distance)
                                if inundated_oids is None: # Error during pre-calc?
                                     # print(f"Warning: Missing pre-calculated OIDs for distance {distance}. Skipping accuracy for this distance.")
                                     continue # Skip this distance if OIDs weren't generated

                                is_inundated_proxy = oid in inundated_oids
                                ct = contingency_tables[distance] # Get the dict for this distance

                                if is_inundated_proxy:
                                    total_inundated_counts[distance] += 1
                                    count_stats[distance]['inundated_counts'].append(count)
                                    if selected_by_model:
                                        ct['tp'] += 1
                                    else:
                                        ct['fn'] += 1
                                else:
                                    total_non_inundated_counts[distance] += 1
                                    count_stats[distance]['non_inundated_counts'].append(count)
                                    if selected_by_model:
                                        ct['fp'] += 1
                                    else:
                                        ct['tn'] += 1
                    else: # No buildings in analysis layer
                         print("  Skipping accuracy calculation as analysis layer was empty.")


                    # --- Report Accuracy for Each Buffer Distance (using calculated values) ---
                    for distance in buffer_distances:
                         try:
                             if total_buildings_in_analysis_layer == 0:
                                 # Report that accuracy couldn't be calculated
                                 report_file.write(f"    --- Buffer: {distance} {buffer_units} ---\n")
                                 report_file.write("    Accuracy calculation skipped: No buildings in the analysis layer.\n\n")
                                 continue

                             ct = contingency_tables[distance]
                             tp, fp, fn, tn = ct['tp'], ct['fp'], ct['fn'], ct['tn']

                             # Check if OIDs were calculated for this distance
                             if inundated_oids_by_distance.get(distance) is None:
                                  report_file.write(f"    --- Buffer: {distance} {buffer_units} ---\n")
                                  report_file.write("    Accuracy calculation skipped: Error during pre-calculation phase for this buffer distance.\n\n")
                                  continue

                             stats_inundated = calculate_stats(count_stats[distance]['inundated_counts'])
                             stats_non_inundated = calculate_stats(count_stats[distance]['non_inundated_counts'])
                             # Use building_count_total_for_acc which reflects buildings actually processed
                             metrics = calculate_accuracy_metrics(tp, fp, fn, tn, building_count_total_for_acc)

                             report_file.write(f"    --- Buffer: {distance} {buffer_units} ---\n")
                             report_file.write(f"      Intersecting Buildings ('Inundated'): {total_inundated_counts[distance]}\n")
                             report_file.write(f"      Non-Intersecting Buildings ('Non-Inundated'): {total_non_inundated_counts[distance]}\n")
                             report_file.write(f"      Stats (Intersecting - Mean/Median/Max Count): {stats_inundated['mean']} / {stats_inundated['median']} / {stats_inundated['max']}\n")
                             report_file.write(f"      Stats (Non-Intersecting - Mean/Median/Max Count): {stats_non_inundated['mean']} / {stats_non_inundated['median']} / {stats_non_inundated['max']}\n")
                             report_file.write(f"      TP: {tp} | FP: {fp} | FN: {fn} | TN: {tn}\n")
                             report_file.write(f"      Sensitivity (Recall): {metrics['sensitivity']}\n")
                             report_file.write(f"      Specificity:          {metrics['specificity']}\n")
                             report_file.write(f"      Precision:            {metrics['precision']}\n")
                             report_file.write(f"      False Positive Rate:  {metrics['fpr']}\n")
                             report_file.write(f"      Accuracy:             {metrics['accuracy']}\n")
                             report_file.write(f"      F1-Score:             {metrics['f1_score']}\n\n")

                         except Exception as report_acc_err:
                              print(f"    ERROR reporting accuracy for buffer {distance}: {report_acc_err}")
                              report_file.write(f"    --- Buffer: {distance} {buffer_units} ---\nERROR generating report: {report_acc_err}\n\n")


                    print(f"  Finished Accuracy Assessment calculation for {n_iter} iterations.")
                    report_file.write("="*25 + f" End Results for N_Iterations = {n_iter} " + "="*25 + "\n\n")

                except Exception as iter_err:
                    print(f"\n--- ERROR during processing for {n_iter} iterations: {iter_err}")
                    report_file.write(f"\n*** ERROR occurred during processing for {n_iter} iterations: {iter_err} ***\n\n")
                    import traceback; traceback.print_exc()
                    print(f"--- Skipping to next iteration value ---")
                    # Remove results for this failed iteration if needed
                    if n_iter in all_results_aggregated:
                        del all_results_aggregated[n_iter]
                    continue # Skip to next n_iter

                iteration_end_time = time.time()
                print(f"===== Finished Run for {n_iter} Iterations in {iteration_end_time - iteration_start_time:.2f} seconds =====")
            # --- End Outer Loop ---

            # 7. Write Aggregated Output Table (after all iterations)
            if all_results_aggregated: # Check if there are any results to write
                 try:
                     output_table_path = write_output_table(config, all_results_aggregated, buildings_data)
                     report_file.write("\n" + "="*70 + "\n")
                     report_file.write("Final Output Table Generation:\n")
                     report_file.write(f"  Aggregated counts written to table: {output_table_path}\n")
                     report_file.write("="*70 + "\n")
                     print(f"\nAggregated results table created at: {output_table_path}")
                 except Exception as table_err:
                     print(f"\n--- ERROR writing final output table: {table_err} ---")
                     report_file.write("\n" + "="*70 + "\n")
                     report_file.write("Final Output Table Generation FAILED:\n")
                     report_file.write(f"  Error: {table_err}\n")
                     report_file.write("="*70 + "\n")
            else:
                 print("\nNo results were generated, skipping final output table creation.")
                 report_file.write("\n" + "="*70 + "\n")
                 report_file.write("Final Output Table Generation Skipped: No results generated during iterations.\n")
                 report_file.write("="*70 + "\n")


        overall_end_time = time.time()
        print(f"\nAll pipeline iterations finished successfully in {overall_end_time - overall_start_time:.2f} seconds.")
        print(f"Comprehensive report saved to: {final_report_path}")

    except ValueError as ve: print(f"\n--- CONFIGURATION OR DATA ERROR ---\n{ve}")
    except arcpy.ExecuteError: print("\n--- ARCPY GEOPROCESSING ERROR ---"); print(arcpy.GetMessages(2))
    except Exception as e: print(f"\n--- UNEXPECTED SCRIPT ERROR ---\n{e}"); import traceback; traceback.print_exc()

    finally:
        # --- Final Cleanup ---
        print("\nStarting final cleanup...")
        # Delete temporary buffers
        print(f"  Attempting cleanup of {len(created_buffers)} temporary buffer file(s)...")
        for buf_path in created_buffers:
            try:
                if arcpy.Exists(buf_path):
                    print(f"    Deleting: {buf_path}")
                    arcpy.management.Delete(buf_path)
                # else: # Optional: Log if buffer already gone
                #     print(f"    Buffer already deleted or never created: {buf_path}")
            except Exception as del_err:
                print(f"    Warning: Could not delete temp buffer {os.path.basename(buf_path)}. Error: {del_err}")

        # Delete temporary residential layer
        # ...(Cleanup logic for analysis_layer / residential_layer_name as before)...
        if 'analysis_layer' in locals() and analysis_layer is not None and analysis_layer != config["buildings"]["path"] and isinstance(analysis_layer, str):
             print(f"  Attempting cleanup of temporary residential layer: {analysis_layer}")
             try:
                 layer_exists = False
                 try:
                     layer_exists = arcpy.Exists(analysis_layer)
                 except AttributeError as exists_attr_err:
                     print(f"    Warning: Attribute error checking existence of temp layer '{analysis_layer}'. Error: {exists_attr_err}. Assuming it might not exist or is invalid.")
                 except Exception as exists_err:
                     print(f"    Warning: Unexpected error checking existence of temp layer '{analysis_layer}'. Error: {exists_err}.")

                 if layer_exists:
                      print(f"    Deleting temporary layer: {analysis_layer}")
                      arcpy.management.Delete(analysis_layer)
                      print(f"    Successfully deleted temp layer: {analysis_layer}")
                 else:
                      print(f"    Temp layer '{analysis_layer}' not found or existence check failed, no deletion performed.")

             except arcpy.ExecuteError as del_exec_err:
                 print(f"    Warning: ArcPy error deleting temp layer {analysis_layer}. Error: {del_exec_err}")
                 print(arcpy.GetMessages(2))
             except Exception as del_err:
                 print(f"    Warning: Unexpected error deleting temp layer {analysis_layer}. Error: {del_err}")
        elif 'analysis_layer' in locals() and analysis_layer == config["buildings"]["path"]:
             print(f"  Cleanup: No temporary residential layer was created (used original path).")
        else:
             print(f"  Cleanup: Temporary residential layer variable not defined or invalid.")

        print("Cleanup finished.")
