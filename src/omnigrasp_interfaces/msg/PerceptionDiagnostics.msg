# PerceptionDiagnostics.msg — Pipeline health published every frame

std_msgs/Header header

string status

float32 gdino_confidence
float32 owlvit_confidence

float32 fusion_iou
string fusion_status

bool depth_valid
float32 depth_value

float32 inference_latency_ms
float32 kalman_innovation
int32 frames_since_last_detection