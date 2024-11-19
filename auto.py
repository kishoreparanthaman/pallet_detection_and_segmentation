from ultralytics.data.annotator import auto_annotate

auto_annotate(data="/home/kishore/peer_ros2/src/yolo/data/valid/images",det_model="/home/kishore/peer_ros2/src/yolo/runs/detect/train19/weights/best.pt",sam_model="sam2_b.pt")