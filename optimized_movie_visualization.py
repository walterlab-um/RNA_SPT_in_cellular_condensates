#!/usr/bin/env python3
"""
Optimized Multi-Channel Movie Visualization with Particle Tracking
================================================================

This script provides enhanced visualization of two-channel imaging data with 
particle tracking overlays, optimized for scientific presentation and analysis.

Key Improvements:
- Adaptive intensity scaling using percentiles
- Enhanced colormaps for better contrast
- Proper scale bar implementation
- Temporal color coding for tracks
- Improved contour visualization
- Dynamic frame selection options
- Better text overlays and annotations

Author: Optimized for scientific accuracy and visual clarity
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import matplotlib.cm as cm
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from tifffile import imread
from rich.progress import track
from scipy import ndimage
from skimage import filters
import warnings
warnings.filterwarnings('ignore')

class OptimizedMovieVisualizer:
    """Enhanced movie visualization class with scientific optimizations"""
    
    def __init__(self, um_per_pixel=0.117, s_per_frame=0.1):
        self.um_per_pixel = um_per_pixel
        self.s_per_frame = s_per_frame
        
        # Enhanced colormaps for better scientific visualization
        self._setup_colormaps()
        
        # Default parameters (can be adjusted)
        self.scalebar_length_um = 2.0  # More appropriate for cellular imaging
        self.scalebar_thickness = 3
        self.track_fade_frames = 10  # Number of frames for track fading
        
    def _setup_colormaps(self):
        """Setup enhanced colormaps for better contrast and scientific presentation"""
        
        # Enhanced blue colormap for better contrast
        colors_blue = ['#000000', '#1a1a2e', '#16213e', '#0f4c75', '#3282b8', '#6495ed']
        n_bins = 256
        self.cmap_blue = clr.LinearSegmentedColormap.from_list('enhanced_blue', colors_blue, N=n_bins)
        
        # Enhanced red colormap with better transparency handling
        colors_red = [(0.0, 0.0, 0.0, 0.0), (0.8, 0.2, 0.2, 1.0)]
        self.cmap_red = clr.LinearSegmentedColormap.from_list('enhanced_red', colors_red, N=n_bins)
        
        # Temporal colormap for tracks (perceptually uniform)
        self.cmap_temporal = cm.get_cmap('viridis')
        
        # Contour colors for better visibility
        self.contour_colors = ['#ffffff', '#ffff00', '#ff6b6b']  # white, yellow, coral
        
    def adaptive_intensity_scaling(self, image, percentile_low=1, percentile_high=99.5):
        """
        Adaptive intensity scaling based on image statistics
        
        Parameters:
        -----------
        image : ndarray
            Input image
        percentile_low : float
            Lower percentile for scaling
        percentile_high : float  
            Upper percentile for scaling
            
        Returns:
        --------
        vmin, vmax : tuple
            Scaling values for display
        """
        # Remove zeros and extreme outliers for better scaling
        non_zero_pixels = image[image > 0]
        if len(non_zero_pixels) > 0:
            vmin = np.percentile(non_zero_pixels, percentile_low)
            vmax = np.percentile(non_zero_pixels, percentile_high)
        else:
            vmin, vmax = np.min(image), np.max(image)
            
        return vmin, vmax
    
    def enhance_contours(self, contours_data, frame_idx):
        """
        Enhanced contour processing for better visualization
        
        Parameters:
        -----------
        contours_data : DataFrame
            Condensate contour data
        frame_idx : int
            Current frame index
            
        Returns:
        --------
        enhanced_contours : list
            List of enhanced contour coordinates
        """
        frame_contours = contours_data[contours_data["frame"] == frame_idx]["contour_coord"].to_list()
        enhanced_contours = []
        
        for cnt_string in frame_contours:
            try:
                x, y = self._parse_contour_string(cnt_string)
                # Smooth contours slightly for better visualization
                if len(x) > 3:
                    x_smooth = ndimage.gaussian_filter1d(x, sigma=0.5, mode='wrap')
                    y_smooth = ndimage.gaussian_filter1d(y, sigma=0.5, mode='wrap')
                    enhanced_contours.append((x_smooth, y_smooth))
                else:
                    enhanced_contours.append((x, y))
            except Exception as e:
                print(f"Warning: Could not parse contour in frame {frame_idx}: {e}")
                continue
                
        return enhanced_contours
    
    def _parse_contour_string(self, contour_string):
        """Parse contour coordinate string into x,y arrays"""
        # Remove brackets and split by coordinate pairs
        clean_string = contour_string.strip('[]')
        coord_pairs = clean_string.split('], [')
        
        x_coords, y_coords = [], []
        for pair in coord_pairs:
            try:
                # Clean up the coordinate pair
                pair = pair.strip('[]')
                x_str, y_str = pair.split(', ')
                x_coords.append(float(x_str))
                y_coords.append(float(y_str))
            except ValueError:
                continue
                
        return np.array(x_coords), np.array(y_coords)
    
    def create_temporal_tracks(self, df_tracks, current_frame, track_length=15):
        """
        Create temporally color-coded tracks with fading effect
        
        Parameters:
        -----------
        df_tracks : DataFrame
            Tracking data
        current_frame : int
            Current frame number
        track_length : int
            Number of previous frames to show in track
            
        Returns:
        --------
        track_segments : list
            List of track segments with temporal coloring
        """
        track_segments = []
        track_ids = df_tracks["trackID"].unique()
        
        for track_id in track_ids:
            track_data = df_tracks[df_tracks["trackID"] == track_id]
            
            # Get track points up to current frame
            track_history = track_data[track_data["t"] <= current_frame]
            track_history = track_history.sort_values("t")
            
            if len(track_history) < 2:
                continue
                
            # Only show recent history
            recent_track = track_history.tail(track_length)
            
            if len(recent_track) >= 2:
                x_coords = recent_track["x"].values
                y_coords = recent_track["y"].values
                frame_nums = recent_track["t"].values
                
                # Create segments with temporal coloring
                for i in range(len(x_coords) - 1):
                    # Calculate color based on recency (newer = brighter)
                    time_weight = (frame_nums[i+1] - (current_frame - track_length)) / track_length
                    time_weight = np.clip(time_weight, 0, 1)
                    
                    segment = {
                        'x': [x_coords[i], x_coords[i+1]],
                        'y': [y_coords[i], y_coords[i+1]], 
                        'color': self.cmap_temporal(time_weight),
                        'alpha': 0.3 + 0.7 * time_weight,
                        'linewidth': 1 + 2 * time_weight
                    }
                    track_segments.append(segment)
                    
        return track_segments
    
    def add_scale_bar(self, ax, image_shape, position='bottom-right'):
        """
        Add a proper scale bar to the image
        
        Parameters:
        -----------
        ax : matplotlib axis
            Axis to add scale bar to
        image_shape : tuple
            Shape of the image (height, width)
        position : str
            Position of scale bar ('bottom-right', 'bottom-left', etc.)
        """
        # Calculate scale bar length in pixels
        scalebar_length_pixels = self.scalebar_length_um / self.um_per_pixel
        
        # Position the scale bar
        margin = 10  # pixels from edge
        height, width = image_shape
        
        if position == 'bottom-right':
            x_start = width - scalebar_length_pixels - margin
            y_pos = height - margin - self.scalebar_thickness
        elif position == 'bottom-left':
            x_start = margin
            y_pos = height - margin - self.scalebar_thickness
        else:  # default to bottom-right
            x_start = width - scalebar_length_pixels - margin  
            y_pos = height - margin - self.scalebar_thickness
            
        # Create scale bar rectangle
        scalebar = Rectangle(
            (x_start, y_pos), 
            scalebar_length_pixels, 
            self.scalebar_thickness,
            facecolor='white',
            edgecolor='black',
            linewidth=0.5
        )
        ax.add_patch(scalebar)
        
        # Add scale bar label
        ax.text(
            x_start + scalebar_length_pixels/2, 
            y_pos - 8,
            f'{self.scalebar_length_um:.0f} μm',
            ha='center', va='top',
            color='white',
            fontsize=10,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7)
        )
    
    def create_enhanced_frame(self, video, df_tracks, df_condensates, frame_idx, 
                            save_path=None, show_tracks=True, show_condensates=True):
        """
        Create an enhanced frame with all optimizations
        
        Parameters:
        -----------
        video : ndarray
            4D video array [frame, channel, height, width]
        df_tracks : DataFrame
            Particle tracking data
        df_condensates : DataFrame
            Condensate segmentation data
        frame_idx : int
            Frame index to visualize
        save_path : str, optional
            Path to save the frame
        show_tracks : bool
            Whether to show particle tracks
        show_condensates : bool
            Whether to show condensate contours
        """
        # Extract channels
        img_blue = video[frame_idx, 1, :, :]  # Assuming channel 1 is blue
        img_red = video[frame_idx, 0, :, :]   # Assuming channel 0 is red
        
        # Create figure with proper sizing
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Adaptive intensity scaling
        blue_vmin, blue_vmax = self.adaptive_intensity_scaling(img_blue, 2, 98)
        red_vmin, red_vmax = self.adaptive_intensity_scaling(img_red, 1, 95)
        
        # Display blue channel
        im1 = ax.imshow(
            img_blue,
            cmap=self.cmap_blue,
            vmin=blue_vmin,
            vmax=blue_vmax,
            aspect='equal'
        )
        
        # Overlay red channel with transparency
        im2 = ax.imshow(
            img_red,
            cmap=self.cmap_red,
            vmin=red_vmin,
            vmax=red_vmax,
            alpha=0.8,
            aspect='equal'
        )
        
        # Add condensate contours if available and requested
        if show_condensates and df_condensates is not None:
            try:
                enhanced_contours = self.enhance_contours(df_condensates, frame_idx)
                for i, (x, y) in enumerate(enhanced_contours):
                    color = self.contour_colors[i % len(self.contour_colors)]
                    ax.plot(x, y, color=color, linewidth=2.5, alpha=0.9)
                    # Close the contour
                    if len(x) > 0:
                        ax.plot([x[-1], x[0]], [y[-1], y[0]], color=color, linewidth=2.5, alpha=0.9)
            except Exception as e:
                print(f"Warning: Could not draw contours for frame {frame_idx}: {e}")
        
        # Add particle tracks if requested
        if show_tracks and df_tracks is not None:
            try:
                # Show temporal tracks
                track_segments = self.create_temporal_tracks(df_tracks, frame_idx)
                for segment in track_segments:
                    ax.plot(
                        segment['x'], segment['y'],
                        color=segment['color'],
                        alpha=segment['alpha'], 
                        linewidth=segment['linewidth']
                    )
                
                # Mark current particle positions
                current_particles = df_tracks[df_tracks["t"] == frame_idx]
                if len(current_particles) > 0:
                    ax.scatter(
                        current_particles["x"], current_particles["y"],
                        marker='o', s=80, 
                        facecolors='none', 
                        edgecolors='yellow',
                        linewidths=2,
                        alpha=0.9
                    )
            except Exception as e:
                print(f"Warning: Could not draw tracks for frame {frame_idx}: {e}")
        
        # Add scale bar
        self.add_scale_bar(ax, img_blue.shape)
        
        # Enhanced time stamp
        time_seconds = frame_idx * self.s_per_frame
        ax.text(
            10, 20, 
            f'Frame: {frame_idx+1:03d}\nTime: {time_seconds:.2f} s',
            fontsize=12,
            color='white',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8)
        )
        
        # Set proper limits and remove axes
        ax.set_xlim(0, img_blue.shape[1])
        ax.set_ylim(img_blue.shape[0], 0)  # Invert y-axis for image coordinates
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Tight layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(
                save_path,
                dpi=300,
                bbox_inches='tight',
                facecolor='black',
                edgecolor='none'
            )
        
        return fig, ax
    
    def create_movie_montage(self, video_path, tracks_path, condensates_path, 
                           output_dir, n_frames=10, frame_selection='adaptive'):
        """
        Create an optimized movie montage
        
        Parameters:
        -----------
        video_path : str
            Path to the TIFF video file
        tracks_path : str
            Path to the tracks CSV file
        condensates_path : str
            Path to the condensates CSV file
        output_dir : str
            Output directory for saved frames
        n_frames : int
            Number of frames to extract
        frame_selection : str
            'linear', 'adaptive', or 'key_events'
        """
        # Load data
        print("Loading video and data files...")
        video = imread(video_path)
        df_tracks = pd.read_csv(tracks_path) if tracks_path else None
        df_condensates = pd.read_csv(condensates_path) if condensates_path else None
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Select frames based on strategy
        total_frames = video.shape[0]
        
        if frame_selection == 'linear':
            frame_indices = np.linspace(0, total_frames-1, n_frames, dtype=int)
        elif frame_selection == 'adaptive':
            # Sample more frames from regions with more particle activity
            if df_tracks is not None:
                activity_per_frame = df_tracks.groupby('t').size()
                # Weight frame selection by activity
                weights = np.ones(total_frames)
                weights[activity_per_frame.index] = activity_per_frame.values
                weights = weights / weights.sum()
                
                frame_indices = np.random.choice(
                    total_frames, size=n_frames, replace=False, p=weights
                )
                frame_indices = np.sort(frame_indices)
            else:
                frame_indices = np.linspace(0, total_frames-1, n_frames, dtype=int)
        else:  # default to linear
            frame_indices = np.linspace(0, total_frames-1, n_frames, dtype=int)
        
        print(f"Creating enhanced montage with {n_frames} frames...")
        
        # Generate enhanced frames
        for i, frame_idx in enumerate(track(frame_indices)):
            save_path = os.path.join(output_dir, f'enhanced_frame_{i+1:03d}.png')
            
            try:
                fig, ax = self.create_enhanced_frame(
                    video, df_tracks, df_condensates, frame_idx, save_path
                )
                plt.close(fig)  # Prevent memory buildup
                
            except Exception as e:
                print(f"Error processing frame {frame_idx}: {e}")
                continue
        
        print(f"Enhanced montage saved to: {output_dir}")

# Example usage and configuration
def main():
    """Main function demonstrating the enhanced visualization"""
    
    # Initialize the enhanced visualizer
    visualizer = OptimizedMovieVisualizer(
        um_per_pixel=0.117,  # Adjust to your imaging parameters
        s_per_frame=0.1      # Adjust to your timing parameters  
    )
    
    # Example paths (adjust to your file locations)
    video_path = "your_video.tif"
    tracks_path = "tracks.csv" 
    condensates_path = "condensates.csv"
    output_dir = "enhanced_montage"
    
    # Create enhanced montage
    visualizer.create_movie_montage(
        video_path=video_path,
        tracks_path=tracks_path, 
        condensates_path=condensates_path,
        output_dir=output_dir,
        n_frames=10,
        frame_selection='adaptive'  # Try 'linear' or 'adaptive'
    )

if __name__ == "__main__":
    main()