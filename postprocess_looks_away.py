import json
import glob
from pathlib import Path
from collections import defaultdict

def process_json_file(file_path):
    """Process a single JSON file and calculate statistics per video."""
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Group results by video
    video_stats = defaultdict(lambda: {
        'player_visible': {'correct': 0, 'total': 0},
        'player_position_final': {'correct': 0, 'total': 0}
    })
    
    for result in data['results']:
        video = result['video']
        query_type = result['metadata']['query_type']
        correct = result['correct']
        
        if query_type == 'player_visible':
            video_stats[video]['player_visible']['total'] += 1
            if correct:
                video_stats[video]['player_visible']['correct'] += 1
        elif query_type == 'player_position_final':
            video_stats[video]['player_position_final']['total'] += 1
            if correct:
                video_stats[video]['player_position_final']['correct'] += 1
    
    # Print statistics
    print(f"\n{'='*80}")
    print(f"File: {Path(file_path).name}")
    print(f"{'='*80}\n")
    
    both_correct_count = 0
    total_videos = len(video_stats)
    
    for video in sorted(video_stats.keys()):
        stats = video_stats[video]
        pv_correct = stats['player_visible']['correct']
        pv_total = stats['player_visible']['total']
        ppf_correct = stats['player_position_final']['correct']
        ppf_total = stats['player_position_final']['total']
        
        both_correct = (pv_correct > 0 and ppf_correct > 0)
        if both_correct:
            both_correct_count += 1
        
        print(f"Video: {video}")
        print(f"  player_visible: {pv_correct}/{pv_total} correct")
        print(f"  player_position_final: {ppf_correct}/{ppf_total} correct")
        print(f"  Both correct: {'Yes' if both_correct else 'No'}")
        print()
    
    # Summary statistics
    total_pv_correct = sum(v['player_visible']['correct'] for v in video_stats.values())
    total_pv_total = sum(v['player_visible']['total'] for v in video_stats.values())
    total_ppf_correct = sum(v['player_position_final']['correct'] for v in video_stats.values())
    total_ppf_total = sum(v['player_position_final']['total'] for v in video_stats.values())
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total videos: {total_videos}")
    print(f"Videos with both query types correct: {both_correct_count}/{total_videos} ({both_correct_count/total_videos*100:.1f}%)")
    print(f"\nOverall player_visible: {total_pv_correct}/{total_pv_total} ({total_pv_correct/total_pv_total*100:.1f}%)")
    print(f"Overall player_position_final: {total_ppf_correct}/{total_ppf_total} ({total_ppf_correct/total_ppf_total*100:.1f}%)")
    
    # Add processed statistics to the data
    data['video_statistics'] = {
        video: {
            'player_visible_correct': stats['player_visible']['correct'],
            'player_visible_total': stats['player_visible']['total'],
            'player_position_final_correct': stats['player_position_final']['correct'],
            'player_position_final_total': stats['player_position_final']['total'],
            'both_correct': (stats['player_visible']['correct'] > 0 and 
                           stats['player_position_final']['correct'] > 0)
        }
        for video, stats in video_stats.items()
    }
    
    data['summary'] = {
        'total_videos': total_videos,
        'videos_with_both_correct': both_correct_count,
        'total_player_visible_correct': total_pv_correct,
        'total_player_visible_total': total_pv_total,
        'total_player_position_final_correct': total_ppf_correct,
        'total_player_position_final_total': total_ppf_total
    }
    
    return data

def main():
    # Find all JSON files ending in "one_looks_away_fixed.json"
    pattern = "results_json/generated/*one_looks_away_fixed.json"
    json_files = glob.glob(pattern)
    
    if not json_files:
        print(f"No files found matching pattern: {pattern}")
        return
    
    print(f"Found {len(json_files)} file(s) to process:")
    for f in json_files:
        print(f"  - {f}")
    
    # Process each file
    for file_path in json_files:
        processed_data = process_json_file(file_path)
        
        # Save processed file
        path = Path(file_path)
        new_filename = path.stem + "-postprocessed" + path.suffix
        output_path = path.parent / new_filename
        
        with open(output_path, 'w') as f:
            json.dump(processed_data, f, indent=2)
        
        print(f"\nSaved processed file to: {output_path}")

if __name__ == "__main__":
    main()



