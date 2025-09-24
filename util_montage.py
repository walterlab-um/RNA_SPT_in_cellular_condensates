from util import *
from param import *

# INDIVIDUAL EXPERIMENT RECONSTRUCTIONS: Each experiment as separate plot
print("🎯 Creating individual final reconstructions for each experiment...")

def analyze_interactions_all_experiments(df_tracks, df_condensates, proximity_threshold=1):
    """Analyze track-condensate interactions across all experiments"""
    
    def get_contour_coords(df_condensates, frame, experiment):
        """Get contour coordinates for specific frame and experiment"""
        subset = df_condensates[
            (df_condensates["frame"] == frame) & 
            (df_condensates["experiment"] == experiment)
        ]
        contours = subset["contour_coord"].to_list()
        parsed_contours = []
        
        for cnt in contours:
            x, y = parse_contour_string(cnt)
            if len(x) > 0:
                parsed_contours.append((np.array(x), np.array(y)))
        return parsed_contours
    
    def is_point_near_boundary(x, y, boundaries, threshold):
        """Check if point is within threshold of any boundary"""
        for cx, cy in boundaries:
            distances = np.sqrt((cx - x)**2 + (cy - y)**2)
            if np.any(distances <= threshold):
                return True
        return False
    
    # Analyze each experiment separately
    experiment_results = {}
    
    print("🔍 Analyzing interactions for each experiment:")
    
    for experiment in df_tracks['experiment'].unique():
        print(f"  Processing: {experiment}")
        
        exp_tracks = df_tracks[df_tracks['experiment'] == experiment]
        exp_condensates = df_condensates[df_condensates['experiment'] == experiment]
        
        if exp_tracks.empty or exp_condensates.empty:
            print(f"    ⚠️ Skipping {experiment} - missing data")
            continue
        
        final_frame = int(exp_tracks['t'].max())
        track_ids = exp_tracks['trackID'].unique()
        
        # Get condensate boundaries for final frame
        contour_boundaries = get_contour_coords(exp_condensates, final_frame, experiment)
        
        if not contour_boundaries:
            print(f"    ⚠️ No condensates found for {experiment} at final frame")
            continue
        
        # Find interacting tracks
        interacting_track_ids = set()
        
        for track_id in track_ids:
            track_data = exp_tracks[
                (exp_tracks['trackID'] == track_id) & 
                (exp_tracks['t'] <= final_frame)
            ].sort_values('t')
            
            # Check if any point in track is near a boundary
            for _, point in track_data.iterrows():
                if is_point_near_boundary(point['x'], point['y'], contour_boundaries, proximity_threshold):
                    interacting_track_ids.add(track_id)
                    break
        
        experiment_results[experiment] = {
            'final_frame': final_frame,
            'contour_boundaries': contour_boundaries,
            'interacting_tracks': interacting_track_ids,
            'total_tracks': len(track_ids),
            'total_condensates': len(contour_boundaries)
        }
        
        print(f"    ✅ {len(interacting_track_ids)}/{len(track_ids)} tracks interacting "
              f"with {len(contour_boundaries)} condensates")
    
    return experiment_results

def create_individual_experiment_reconstructions(df_tracks,
                                                 df_condensates,
                                                 experiment_results,
                                                 folder_path):
    """Create individual final reconstruction plots for each experiment"""
    
    n_experiments = len(experiment_results)
    if n_experiments == 0:
        print("❌ No experiments to plot!")
        return
    
    print(f"📊 Creating {n_experiments} individual reconstruction plots...")
    
    # Create individual plot for each experiment
    for experiment, results in track(experiment_results.items(), description="Creating reconstructions"):
        
        # Create single figure for this experiment
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.set_facecolor('white')
        
        # Get experiment-specific data
        exp_tracks = df_tracks[df_tracks['experiment'] == experiment]
        final_frame = results['final_frame']
        contour_boundaries = results['contour_boundaries']
        interacting_tracks = results['interacting_tracks']
        
        # Plot condensate boundaries
        for cx, cy in contour_boundaries:
            ax.plot(cx, cy, lw=3, c="#2E86AB", alpha=0.8)  # Professional blue
            ax.plot([cx[-1], cx[0]], [cy[-1], cy[0]], c="#2E86AB", lw=3, alpha=0.8)
        
        # Plot ONLY interacting tracks
        for track_id in interacting_tracks:
            full_track_data = exp_tracks[
                (exp_tracks['trackID'] == track_id) & 
                (exp_tracks['t'] <= final_frame)
            ].sort_values('t')
            
            if len(full_track_data) > 1:
                ax.plot(
                    full_track_data['x'], full_track_data['y'],
                    color='#F24236',  # Professional red
                    alpha=0.7,
                    linewidth=2
                )
        
        # Add experiment info (larger for individual plot)
        time_seconds = final_frame * s_per_frame
        info_text = (f'{experiment}\n'
                    f'{time_seconds:.1f}s\n'
                    f'Interacting: {len(interacting_tracks)}\n'
                    f'Total Tracks: {results["total_tracks"]}\n'
                    f'Condensates: {len(contour_boundaries)}')
        
        ax.text(
            10, 20,
            info_text,
            fontsize=16,
            color='black',
            fontweight='bold',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9)
        )
        
        # Add scale bar
        if not exp_tracks.empty:
            x_range = exp_tracks['x'].max() - exp_tracks['x'].min()
            y_range = exp_tracks['y'].max() - exp_tracks['y'].min()
            
            scalebar_length_pixels = 2.0 / um_per_pixel
            margin = 15
            
            scale_x = exp_tracks['x'].max() - scalebar_length_pixels - margin
            scale_y = exp_tracks['y'].max() - margin - 4
            
            ax.add_patch(Rectangle(
                (scale_x, scale_y), scalebar_length_pixels, 4,
                facecolor='black', edgecolor='white', linewidth=1
            ))
            
            ax.text(
                scale_x + scalebar_length_pixels/2, scale_y - 10,
                '2 μm',
                ha='center', va='top', color='black', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
            )
            
            # Set limits with margin
            margin_pixels = max(x_range, y_range) * 0.05
            ax.set_xlim(exp_tracks['x'].min() - margin_pixels, exp_tracks['x'].max() + margin_pixels)
            ax.set_ylim(exp_tracks['y'].max() + margin_pixels, exp_tracks['y'].min() - margin_pixels)
        
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add title
        ax.set_title(
            f'RNA-Condensate Interactions: {experiment}',
            fontsize=18, fontweight='bold', pad=20
        )
        
        plt.tight_layout()
        
        # Save individual plot with clean filename
        safe_filename = experiment.replace(" ", "_").replace("/", "_").replace("\\", "_")
        output_path = join(folder_path, f"reconstruction_{safe_filename}.png")

        plt.savefig(
            output_path,
            format="png",
            bbox_inches="tight",
            dpi=300,
            facecolor='white'
        )
        plt.show()
        plt.close()
        
        print(f"  ✅ Saved: reconstruction_{safe_filename}.png")
    
    print(f"🎉 All {n_experiments} individual reconstructions created!")

# INDIVIDUAL EXPERIMENT CONDENSATE MONTAGES: One montage per experiment (with size limits)
print("🔍 Creating individual condensate montages for each experiment...")

def create_individual_experiment_montages(df_tracks, df_condensates, experiment_results, 
                                        zoom_margin=25, montage_cols=4):
    """Create individual condensate montages for each experiment with automatic splitting for large datasets"""
    
    if not experiment_results:
        print("❌ No experiment results available!")
        return
    
    # Calculate safe limits to avoid matplotlib size errors
    max_pixels = 2**16  # Maximum image dimension
    subplot_height_pixels = 4 * 300  # 4 inches * 300 DPI
    max_safe_rows = (max_pixels // subplot_height_pixels) - 2  # Safety margin
    max_condensates_per_montage = max_safe_rows * montage_cols
    
    print(f"📊 Creating condensate montages for {len(experiment_results)} experiments...")
    print(f"   Maximum condensates per montage: {max_condensates_per_montage}")
    
    # Track colors for variety
    track_colors = ['#F24236', '#E66100', '#A63603', '#D95F02', '#CC79A7', 
                   '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00']
    
    # Process each experiment individually
    for experiment, results in track(experiment_results.items(), description="Creating montages"):
        
        print(f"  Processing montage for: {experiment}")
        
        exp_tracks = df_tracks[df_tracks['experiment'] == experiment]
        final_frame = results['final_frame']
        contour_boundaries = results['contour_boundaries']
        interacting_tracks = results['interacting_tracks']
        
        # Collect condensates with interactions for this experiment
        experiment_condensate_data = []
        
        for condensate_idx, (cx, cy) in enumerate(contour_boundaries):
            # Find tracks that interact with this specific condensate
            condensate_interacting_tracks = []
            
            for track_id in interacting_tracks:
                track_data = exp_tracks[
                    (exp_tracks['trackID'] == track_id) & 
                    (exp_tracks['t'] <= final_frame)
                ].sort_values('t')
                
                # Check if track interacts with THIS condensate
                track_interacts = False
                for _, point in track_data.iterrows():
                    distances = np.sqrt((cx - point['x'])**2 + (cy - point['y'])**2)
                    if np.any(distances <= proximity_threshold):
                        track_interacts = True
                        break
                
                if track_interacts and len(track_data) > 1:
                    condensate_interacting_tracks.append(track_data)
            
            # Only include condensates with interacting tracks
            if condensate_interacting_tracks:
                experiment_condensate_data.append({
                    'condensate_idx': condensate_idx,
                    'contour': (cx, cy),
                    'interacting_tracks': condensate_interacting_tracks
                })
        
        if not experiment_condensate_data:
            print(f"    ⚠️ No condensates with interacting tracks found for {experiment}")
            continue
        
        n_condensates = len(experiment_condensate_data)
        print(f"    Found {n_condensates} condensates with interactions")
        
        # Split into multiple montages if needed
        if n_condensates > max_condensates_per_montage:
            n_montages = (n_condensates + max_condensates_per_montage - 1) // max_condensates_per_montage
            print(f"    ⚠️ Creating {n_montages} separate montages (too many condensates for single image)")
        else:
            n_montages = 1
        
        # Create montages
        for montage_idx in range(n_montages):
            start_idx = montage_idx * max_condensates_per_montage
            end_idx = min(start_idx + max_condensates_per_montage, n_condensates)
            montage_condensates = experiment_condensate_data[start_idx:end_idx]
            
            n_condensates_this_montage = len(montage_condensates)
            
            if n_montages > 1:
                print(f"      Creating montage {montage_idx + 1}/{n_montages} with {n_condensates_this_montage} condensates")
            
            # Calculate montage layout for this subset
            montage_rows = (n_condensates_this_montage + montage_cols - 1) // montage_cols
            
            # Create montage figure
            fig_width = montage_cols * 4
            fig_height = montage_rows * 4
            
            # Additional safety check
            if fig_height * 300 > max_pixels:
                # Further reduce if still too large
                safe_height = (max_pixels // 300) - 1
                fig_height = safe_height
                print(f"      ⚠️ Reducing figure height to {fig_height} inches for safety")
            
            try:
                fig, axes = plt.subplots(
                    montage_rows, montage_cols, 
                    figsize=(fig_width, fig_height)
                )
                
                # Handle subplot configuration
                if n_condensates_this_montage == 1:
                    axes = [axes]
                elif montage_rows == 1:
                    axes = axes.reshape(1, -1)
                elif montage_cols == 1:
                    axes = axes.reshape(-1, 1)
                
                axes_flat = axes.flatten()
                
                # Plot each condensate in this montage
                for plot_idx, condensate_info in enumerate(montage_condensates):
                    ax = axes_flat[plot_idx]
                    ax.set_facecolor('white')
                    
                    cx, cy = condensate_info['contour']
                    tracks = condensate_info['interacting_tracks']
                    condensate_idx = condensate_info['condensate_idx']
                    
                    # Calculate zoom window including all track points
                    min_x, max_x = np.min(cx), np.max(cx)
                    min_y, max_y = np.min(cy), np.max(cy)
                    
                    # Expand window to include track endpoints
                    for track_data in tracks:
                        min_x = min(min_x, track_data['x'].min())
                        max_x = max(max_x, track_data['x'].max())
                        min_y = min(min_y, track_data['y'].min())
                        max_y = max(max_y, track_data['y'].max())
                    
                    zoom_xmin = int(min_x - zoom_margin)
                    zoom_xmax = int(max_x + zoom_margin)
                    zoom_ymin = int(min_y - zoom_margin)
                    zoom_ymax = int(max_y + zoom_margin)
                    
                    # Plot condensate boundary
                    ax.plot(cx, cy, lw=4, c="#2E86AB", alpha=0.9)
                    ax.plot([cx[-1], cx[0]], [cy[-1], cy[0]], c="#2E86AB", lw=4, alpha=0.9)
                    
                    # Plot interacting tracks with different colors
                    for track_idx, track_data in enumerate(tracks):
                        color = track_colors[track_idx % len(track_colors)]
                        
                        ax.plot(
                            track_data['x'], track_data['y'],
                            color=color,
                            alpha=0.8,
                            linewidth=3,
                            label=f'Track {int(track_data.iloc[0]["trackID"])}'
                        )
                        
                        # Mark start and end points
                        ax.plot(
                            track_data.iloc[0]['x'], track_data.iloc[0]['y'],
                            marker='o', markersize=6, color=color,
                            markerfacecolor='white', markeredgewidth=2
                        )
                        ax.plot(
                            track_data.iloc[-1]['x'], track_data.iloc[-1]['y'],
                            marker='s', markersize=5, color=color,
                            markerfacecolor=color, markeredgewidth=1, alpha=0.8
                        )
                    
                    # # Add condensate label in center
                    # center_x, center_y = np.mean(cx), np.mean(cy)
                    # ax.text(
                    #     center_x, center_y,
                    #     f'C{condensate_idx + 1}',
                    #     fontsize=12,
                    #     color='white',
                    #     fontweight='bold',
                    #     ha='center', va='center',
                    #     bbox=dict(boxstyle='circle,pad=0.3', facecolor="#2E86AB", alpha=0.8)
                    # )
                    
                    # Add scale bar (1 μm for zoomed view)
                    scalebar_length_pixels = 1.0 / um_per_pixel
                    scale_x = zoom_xmax - scalebar_length_pixels - 5
                    scale_y = zoom_ymax - 8
                    
                    ax.add_patch(Rectangle(
                        (scale_x, scale_y), scalebar_length_pixels, 3,
                        facecolor='black', edgecolor='white', linewidth=1
                    ))
                    
                    ax.text(
                        scale_x + scalebar_length_pixels/2, scale_y - 5,
                        '1 μm',
                        ha='center', va='top', color='black', fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9)
                    )
                    
                    # Set zoom limits
                    ax.set_xlim(zoom_xmin, zoom_xmax)
                    ax.set_ylim(zoom_ymax, zoom_ymin)  # Invert Y for image coordinates
                    ax.set_aspect('equal')
                    ax.axis('off')
                    
                    # Add title with condensate and interaction info
                    ax.set_title(
                        f'Condensate {condensate_idx + 1}\n{len(tracks)} Interacting RNAs',
                        fontsize=11,
                        fontweight='bold',
                        pad=10
                    )
                    
                    # Add legend if multiple tracks but not too many
                    if 1 < len(tracks) <= 4:
                        ax.legend(
                            loc='upper left', 
                            fontsize=8,
                            frameon=True,
                            framealpha=0.8,
                            edgecolor='gray'
                        )
                
                # Hide unused subplots
                for idx in range(n_condensates_this_montage, len(axes_flat)):
                    axes_flat[idx].axis('off')
                
                # Add overall title for this montage
                if n_montages > 1:
                    title_text = (f'Condensate Interactions: {experiment} (Part {montage_idx + 1}/{n_montages})\n'
                                f'{n_condensates_this_montage} of {n_condensates} Total Condensates')
                else:
                    title_text = (f'Condensate Interactions: {experiment}\n'
                                f'{n_condensates_this_montage} Condensates with RNA Interactions')
                
                fig.suptitle(
                    title_text,
                    fontsize=16,
                    fontweight='bold',
                    y=0.98
                )
                
                plt.tight_layout()
                plt.subplots_adjust(top=0.90)
                
                # Save individual montage with clean filename
                safe_filename = experiment.replace(" ", "_").replace("/", "_").replace("\\", "_")
                
                if n_montages > 1:
                    output_path = join(folder_path, f"condensate_montage_{safe_filename}_part{montage_idx + 1:02d}.png")
                else:
                    output_path = join(folder_path, f"condensate_montage_{safe_filename}.png")
                
                plt.savefig(
                    output_path,
                    format="png",
                    bbox_inches="tight",
                    dpi=300,
                    facecolor='white'
                )
                plt.show()
                plt.close()
                
                if n_montages > 1:
                    print(f"      ✅ Saved: condensate_montage_{safe_filename}_part{montage_idx + 1:02d}.png")
                else:
                    print(f"    ✅ Saved: condensate_montage_{safe_filename}.png")
                
            except Exception as e:
                print(f"      ❌ Error creating montage {montage_idx + 1}: {e}")
                plt.close('all')  # Clean up any partial figures
                continue
    
    print(f"🎉 All individual condensate montages created!")