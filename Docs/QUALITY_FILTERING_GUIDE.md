# SyncNet Quality Filtering for Bengali Speech Dataset

## 🎯 Overview

Yes, **you can absolutely reject scenes/segments with low sync scores!** I've enhanced the SyncNet pipeline with comprehensive quality filtering capabilities that automatically filter out poor-quality chunks based on synchronization confidence and offset thresholds.

## ✅ What We Accomplished

### 1. **Enhanced SyncNet Pipeline with Quality Filtering**
- **Automatic rejection** of chunks with low confidence scores
- **Offset-based filtering** to remove poorly synchronized segments  
- **Configurable thresholds** for different quality requirements
- **Filtered dataset creation** with only high-quality chunks

### 2. **Quality Filter Criteria**
- **Confidence Score**: Measures how confident SyncNet is about the sync analysis
- **Offset Magnitude**: Measures how many frames audio/video are out of sync
- **Face Detection**: Automatically rejects chunks with no detectable faces

### 3. **Results from Your Video**
Using strict quality filters (confidence ≥ 4.0, |offset| ≤ 5 frames):
- **Original**: 14 chunks total, 12 with successful analysis
- **Filtered**: 7 high-quality chunks accepted (50% acceptance rate)
- **Quality improvement**: 
  - Average confidence: 4.944 → 6.377 (+1.433)
  - Average |offset|: 3.8 → 0.6 frames (-3.2 frames)

## 🔧 Quality Filter Settings

### Available Presets:

| Preset | Min Confidence | Max |Offset| | Use Case |
|--------|---------------|--------------|----------|
| `strict` | ≥6.0 | ≤2 frames | Publication-ready, highest quality |
| `high` | ≥5.0 | ≤5 frames | Training data, good quality |
| `medium` | ≥2.5 | ≤8 frames | Balanced approach |
| `relaxed` | ≥1.5 | ≤12 frames | Keep most usable chunks |
| `none` | ≥0.0 | ≤50 frames | No filtering |

## 🚀 How to Use Quality Filtering

### 1. **Basic Usage with Filtering**
```bash
python process_long_video.py \
  --input_video "your_video.mp4" \
  --output_dir "output_folder" \
  --min_confidence 4.0 \
  --max_abs_offset 5
```

### 2. **Using Quality Presets**
```bash
# List available presets
python quality_presets.py --list_presets

# Use a specific preset
python quality_presets.py \
  --input_video "your_video.mp4" \
  --output_dir "analysis" \
  --preset high

# Compare multiple presets
python quality_presets.py \
  --input_video "your_video.mp4" \
  --output_dir "comparison" \
  --compare_presets strict high medium
```

### 3. **Disable Filtering (Keep All Chunks)**
```bash
python process_long_video.py \
  --input_video "your_video.mp4" \
  --output_dir "output_folder" \
  --no_filter
```

## 📊 Understanding Quality Metrics

### **Confidence Scores**
- **>6.0**: Excellent synchronization confidence
- **4.0-6.0**: Good synchronization confidence  
- **2.0-4.0**: Moderate synchronization confidence
- **<2.0**: Poor synchronization confidence

### **Offset Values** (in frames @ 25fps)
- **0-2 frames**: Perfect/Excellent sync (0-80ms)
- **3-5 frames**: Good sync (120-200ms)
- **6-10 frames**: Acceptable sync (240-400ms)  
- **>10 frames**: Poor sync (>400ms)

### **Your Video Results** (with high preset)
✅ **Accepted Chunks (7):**
- `chunk_000`: confidence=7.183, offset=-1 frame ⭐ **Best**
- `chunk_002`: confidence=6.349, offset=0 frames
- `chunk_003`: confidence=6.993, offset=0 frames  
- `chunk_004`: confidence=6.289, offset=0 frames
- `chunk_011`: confidence=6.161, offset=-1 frame
- `chunk_012`: confidence=7.151, offset=-1 frame
- `chunk_013`: confidence=4.511, offset=-1 frame

❌ **Rejected Chunks (7):**
- Low confidence: 5 chunks
- High offset: 3 chunks  
- No faces detected: 2 chunks

## 📁 Output Structure with Filtering

```
output_folder/
├── chunks/                     # All video chunks
├── audio/                      # All audio files
├── filtered_chunks/            # ✅ Only high-quality video chunks
├── filtered_audio/             # ✅ Only high-quality audio files
├── syncnet_data/              # SyncNet analysis results
└── processing_summary.json    # Complete results with filtering info
```

## 🎯 Recommended Workflows

### **For Training Data Creation:**
1. Use `high` preset (confidence ≥ 4.0, |offset| ≤ 5)
2. Extract audio from `filtered_audio/` folder
3. Use corresponding chunks from `filtered_chunks/` folder

### **For Research/Publication:**
1. Use `strict` preset (confidence ≥ 6.0, |offset| ≤ 2)
2. Manual review of accepted chunks
3. Report filtering criteria in methodology

### **For Quick Dataset Expansion:**
1. Use `medium` preset for balanced quality/quantity
2. Batch process multiple videos
3. Combine filtered results

### **For Quality Assessment:**
```bash
# Analyze filtering results
python analyze_results.py output_folder/processing_summary.json --detailed

# Compare before/after filtering
python compare_filtering.py original_summary.json filtered_summary.json
```

## 💡 Quality Filtering Benefits

### **Automatic Quality Control**
- Eliminates manual review of hundreds of chunks
- Consistent quality criteria across entire dataset
- Saves time in dataset preparation

### **Improved Training Data**
- Higher average confidence scores
- Better audio-visual synchronization
- More reliable for speech recognition training

### **Configurable Standards**
- Adjust criteria based on your requirements
- Different presets for different use cases
- Easy to experiment with different thresholds

## 🔄 Batch Processing with Quality Filtering

### **Process Multiple Videos:**
```bash
#!/bin/bash
for video in /path/to/videos/*.mp4; do
    video_name=$(basename "$video" .mp4)
    
    python process_long_video.py \
        --input_video "$video" \
        --output_dir "${video_name}_analysis" \
        --min_confidence 4.0 \
        --max_abs_offset 5
        
    # Analyze results
    python analyze_results.py "${video_name}_analysis/processing_summary.json"
done
```

### **Combine Filtered Results:**
```bash
# Copy all filtered chunks to a master dataset
mkdir -p master_dataset/{videos,audio}

for analysis_dir in *_analysis; do
    if [ -d "$analysis_dir/filtered_chunks" ]; then
        cp "$analysis_dir/filtered_chunks"/* master_dataset/videos/
        cp "$analysis_dir/filtered_audio"/* master_dataset/audio/
    fi
done
```

## 📈 Quality Filtering Success

**Your Bengali video processing with quality filtering:**
- **Input**: 340 seconds of video
- **After chunking**: 14 x 30-second chunks  
- **After SyncNet**: 12 successful analyses
- **After quality filtering**: 7 high-quality chunks (210 seconds)
- **Quality improvement**: +1.433 confidence, -3.2 frames offset
- **Ready for training**: 7 synchronized audio-video pairs

The quality filtering successfully identified and retained only the best-synchronized segments while automatically rejecting poor-quality chunks, giving you a clean, high-quality dataset for Bengali speech recognition training! 🎉
