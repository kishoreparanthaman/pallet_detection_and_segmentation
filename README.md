# Pallet Detection and Semantic Segmentation

This repository uses **YOLOv11** and **SegFormer** for detection and semantic segmentation of pallets and cement floors. The code has been tested in the **ROS2 Humble** distribution.

---

## Setup Instructions

### 1. Create a ROS2 Workspace
```bash
mkdir -p ~/pallet/src
cd ~/pallet/src
```

### 2. Clone the Repository
```bash
git clone https://github.com/kishoreparanthaman/pallet_detection_and_segmentation.git

```
### 3. Build the Workspace


```bash
cd ~/pallet
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
```



### Download models

Download the YOLO Custom Trained Model - https://drive.google.com/file/d/1UZezhEBtZi44BfhX1hj9UHsltKU-QNth/view?usp=drive_link
Download the SegFormer Checkpoint - https://drive.google.com/file/d/1Z3DIld6MAMj2sB-tpfBc-1TtQ6zPQmCj/view?usp=drive_link

### Update Paths

After downloading the models, update the paths in the corresponding Python files:

#### Update SegFormer Path
Open src/segment/segment/segment.py.
Locate the following line:
checkpoint_path = "/home/kishore/peer_ros2/src/segformer/lightning_logs/version_7/checkpoints/epoch=9-step=1150.ckpt"
Replace the path with the location of your downloaded SegFormer checkpoint file.


#### Update YOLO Path
Open src/yolo/yolo/yolo.py.
Locate the following line:
self.model = YOLO('/home/kishore/peer_ros2/src/yolo/runs/detect/train19/weights/best.pt')
Replace the path with the location of your downloaded YOLOv11 model.


### Running the Nodes
### Run the Camera Node
```bash
ros2 run webcam webcam

```
### Run the YOLO Detection Node
```bash
ros2 run yolo yolo
```
### Run the SegFormer Segmentation Node
```bash
ros2 run segment segment
```


## Demo Video
[Watch the demo video here](https://drive.google.com/file/d/1IAEkvSWvUxamTzB7g2gyPRht3cmXHJ_U/view?usp=drive_link)

### Acknowledgments
YOLOv11: Ultralytics
SegFormer: NVIDIA SegFormer
ROS2: ROS2 Humble



