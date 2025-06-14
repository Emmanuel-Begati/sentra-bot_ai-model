# Smart Crop Health Monitoring System

## Environment Setup

1. **Install Dependencies**:
```bash
pip install ultralytics fastapi uvicorn torch torchvision opencv-python pillow requests python-multipart pandas numpy pyyaml python-dotenv aiofiles
```

2. **Set Environment Variables**:
Create a `.env` file in the root directory:
```
PLANTNET_API_KEY=your_plantnet_api_key_here
CUDA_AVAILABLE=true  # Set to false if no GPU
```

3. **Project Structure**:
```
pest_model/
├── config.py                 # Configuration settings
├── main.py                   # Main FastAPI application
├── data/                     # Merged dataset
├── models/                   # Trained model weights
├── training/                 # Training scripts
├── utils/                    # Helper utilities
├── raspberry_api/           # Raspberry Pi integration
│   ├── sensor_collector.py  # Sensor data collection
│   ├── camera_client.py     # Camera capture and API client
│   └── deploy.sh            # Raspberry Pi setup script
├── advice/                  # Farming advice system
│   └── rag_advisor.py       # Enhanced advice with RAG
├── logs/                    # System logs
└── dataset/                 # Original datasets
```

## Usage

### Backend Setup
1. **Merge Datasets**: Run the dataset merger to combine all datasets
2. **Train Model**: Execute training script with merged dataset
3. **Start API**: Launch FastAPI backend for predictions
```bash
python main.py
```

### Raspberry Pi Setup
1. **Deploy to Pi**: Copy raspberry_api files to your Raspberry Pi
2. **Run setup script**: `bash deploy.sh` (on the Pi)
3. **Configure**: Edit config.py with your API endpoint
4. **Test**: `python3 camera_client.py --mode single`
5. **Start monitoring**: `python3 camera_client.py --mode continuous --interval 30`

### API Usage
- Single analysis: POST image + sensor data to `/predict`
- Get detailed health: GET `/health/{prediction_id}`
- Get farming advice: GET `/advice/{prediction_id}`

## Features

- **Multi-dataset Training**: Combines multiple plant disease datasets
- **Plant Identification**: Uses PlantNet API for species identification
- **Disease/Pest Detection**: YOLOv8-based detection system
- **Health Scoring**: Comprehensive crop health percentage calculation
- **Enhanced Advice System**: RAG-based farming recommendations
- **Raspberry Pi Integration**: Real-time sensor data collection
- **Continuous Monitoring**: Automated periodic health assessments
- **FastAPI Backend**: RESTful API for all operations

## API Endpoints

- `POST /predict` - Analyze crop image and sensor data
- `POST /identify-plant` - Identify plant species
- `GET /health/{prediction_id}` - Get health analysis
- `GET /advice/{prediction_id}` - Get farming recommendations
- `GET /predictions` - List recent predictions
- `GET /health` - API health check

## Raspberry Pi Components

### Sensors Supported
- **DHT22**: Temperature and humidity
- **Soil moisture sensor**: Analog soil moisture
- **pH sensor**: Soil pH measurement
- **BH1750**: Light intensity sensor

### Camera Options
- **PiCamera2**: Official Raspberry Pi camera
- **USB Camera**: Via OpenCV
- **Simulation mode**: For testing without hardware

### Usage Examples

**Single capture and analysis:**
```bash
python3 camera_client.py --mode single --location "greenhouse_1" --crop-type "tomato"
```

**Continuous monitoring:**
```bash
python3 camera_client.py --mode continuous --interval 30 --location "field_a"
```

**Custom API endpoint:**
```bash
python3 camera_client.py --api-url "http://your-server:8000"
```
# sentra_ai-model
# sentra_ai-model
