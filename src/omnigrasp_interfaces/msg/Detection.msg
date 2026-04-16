# Detection.msg — Output of the detection fusion module
# Represents a single detected object with multi-model confidence data

std_msgs/Header header

# Bounding box in pixels
float32 x_min
float32 y_min
float32 x_max
float32 y_max

# Per-model confidence (0.0 = not detected, up to 1.0)
float32 gdino_confidence
float32 owlvit_confidence

# Fused result
float32 fused_confidence
float32 fusion_iou

# Detection context
string prompt
string fusion_status