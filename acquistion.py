# Intégration de l'algorithme de reconnaissance biométrique dans l'application
def authenticate_user():
    captured_data = capture_biometric_data()
    preprocessed_features = preprocess_and_extract_features(captured_data)
    authentication_result = recognize_user(preprocessed_features, trained_model)

    if authentication_result:
        grant_access()
    else:
        deny_access()
