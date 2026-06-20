import mediapipe as mp


print("mediapipe file:", mp.__file__)
print("mediapipe version:", getattr(mp, "__version__", "unknown"))
print("has solutions:", hasattr(mp, "solutions"))
print("solutions:", mp.solutions)
print("pose:", mp.solutions.pose)
