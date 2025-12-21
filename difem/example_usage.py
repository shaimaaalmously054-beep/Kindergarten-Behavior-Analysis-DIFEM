from difem.feature_extraction import extract_difem_features

features = extract_difem_features(
    "data/sample_video_json"
)

print("DIFEM feature vector:", features)
print("Feature dimension:", features.shape)
