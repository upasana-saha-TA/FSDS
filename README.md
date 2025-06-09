# Median housing value prediction

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data.

The following techniques have been used:

 - Linear regression
 - Decision Tree
 - Random Forest

## Steps performed
 - We prepare and clean the data. We check and impute for missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techinuqies are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.

## To excute the individual scripts
```bash
python < scriptname.py >
```
## To excute the entire flow through main script:
```bash
python scripts/main.py
```

## Docker Deployment

1. **Build the multi-architecture image** (amd64 + arm64):
```bash
docker buildx build \
    --platform linux/amd64,linux/arm64 \
    --no-cache --pull \
    -t <YOUR_ECR_REGISTRY>/fsds_housinglib:latest \
    --push .
```

2. **Deploy to Kubernetes:**
Update housing-app-pod.yaml to reference your image and imagePullPolicy: Always.

```bash
docker buildx build \
    --platform linux/amd64,linux/arm64 \
    --no-cache --pull \
    -t <YOUR_ECR_REGISTRY>/fsds_housinglib:latest \
    --push .
```
3. **Local testing (without Kubernetes) :**
```bash
docker run --rm -it <YOUR_ECR_REGISTRY>/fsds_housinglib:latest \
python -u scripts/inference.py --no_console_log=False
```
## Monitoring Data Drift with Evidently AI
To detect when incoming data drifts from the training distribution, execute the scripts by:
```bash
python scripts/monitoring.py
```