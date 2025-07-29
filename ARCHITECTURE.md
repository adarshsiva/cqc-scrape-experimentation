# CQC Rating Predictor - System Architecture

## Table of Contents
1. [High-Level Architecture](#high-level-architecture)
2. [Data Flow Architecture](#data-flow-architecture)
3. [Component Architecture](#component-architecture)
4. [Security Architecture](#security-architecture)
5. [Deployment Architecture](#deployment-architecture)
6. [Database Schema](#database-schema)
7. [API Architecture](#api-architecture)
8. [ML Pipeline Architecture](#ml-pipeline-architecture)

## High-Level Architecture

### System Overview
```mermaid
graph TB
    subgraph "External Systems"
        CQC[CQC API]
        Users[End Users]
    end
    
    subgraph "GCP - Data Ingestion"
        CS[Cloud Scheduler]
        CF1[Cloud Function<br/>cqc-data-ingestion]
        SM[Secret Manager]
        GCS1[Cloud Storage<br/>Raw Data]
    end
    
    subgraph "GCP - Data Processing"
        DF[Cloud Dataflow<br/>ETL Pipeline]
        BQ[(BigQuery<br/>Data Warehouse)]
        GCS2[Cloud Storage<br/>Temp Files]
    end
    
    subgraph "GCP - ML Platform"
        VAI[Vertex AI<br/>Training]
        VAE[Vertex AI<br/>Endpoint]
        GCS3[Cloud Storage<br/>ML Artifacts]
    end
    
    subgraph "GCP - Serving Layer"
        CF2[Cloud Function<br/>cqc-rating-prediction]
        CM[Cloud Monitoring]
        CL[Cloud Logging]
    end
    
    CS -->|Triggers Weekly| CF1
    CF1 -->|Reads Secrets| SM
    CF1 -->|Fetches Data| CQC
    CF1 -->|Stores JSON| GCS1
    GCS1 -->|Reads Raw Data| DF
    DF -->|Writes Temp| GCS2
    DF -->|Loads Tables| BQ
    BQ -->|Training Data| VAI
    VAI -->|Saves Models| GCS3
    VAI -->|Deploys Model| VAE
    Users -->|Prediction Request| CF2
    CF2 -->|Gets Predictions| VAE
    CF2 -->|Logs Predictions| BQ
    CF1 -->|Logs| CL
    CF2 -->|Logs| CL
    DF -->|Metrics| CM
    VAE -->|Metrics| CM
```

## Data Flow Architecture

### End-to-End Data Pipeline
```mermaid
sequenceDiagram
    participant Scheduler as Cloud Scheduler
    participant Ingestion as Ingestion Function
    participant CQC as CQC API
    participant RawStorage as Raw Storage
    participant Dataflow as Dataflow Pipeline
    participant BigQuery as BigQuery
    participant VertexAI as Vertex AI
    participant Model as Model Endpoint
    participant PredAPI as Prediction API
    participant User as User
    
    Scheduler->>Ingestion: Trigger (Weekly)
    Ingestion->>CQC: GET /providers
    CQC-->>Ingestion: Provider Data
    Ingestion->>CQC: GET /locations
    CQC-->>Ingestion: Location Data
    Ingestion->>RawStorage: Store JSON Files
    
    Note over Dataflow: ETL Process
    Dataflow->>RawStorage: Read Raw Data
    Dataflow->>Dataflow: Transform & Validate
    Dataflow->>BigQuery: Load Clean Data
    
    Note over VertexAI: ML Training
    VertexAI->>BigQuery: Query Training Data
    VertexAI->>VertexAI: Train Models
    VertexAI->>Model: Deploy Best Model
    
    Note over User: Prediction Flow
    User->>PredAPI: POST /predict
    PredAPI->>Model: Get Prediction
    Model-->>PredAPI: Rating Prediction
    PredAPI-->>User: Response
    PredAPI->>BigQuery: Log Prediction
```

## Component Architecture

### Microservices Architecture
```mermaid
graph LR
    subgraph "Ingestion Service"
        ING_API[API Client]
        ING_AUTH[Auth Handler]
        ING_STORE[Storage Writer]
        ING_LOG[Logger]
    end
    
    subgraph "ETL Service"
        ETL_READ[Data Reader]
        ETL_TRANS[Transformer]
        ETL_VAL[Validator]
        ETL_LOAD[BQ Loader]
    end
    
    subgraph "ML Service"
        ML_PREP[Data Prep]
        ML_FE[Feature Engineering]
        ML_TRAIN[Model Training]
        ML_EVAL[Evaluation]
        ML_REG[Model Registry]
    end
    
    subgraph "Prediction Service"
        PRED_API[REST API]
        PRED_VAL[Input Validator]
        PRED_CLIENT[Model Client]
        PRED_CACHE[Cache Layer]
        PRED_LOG[Prediction Logger]
    end
    
    ING_API --> ING_AUTH
    ING_AUTH --> ING_STORE
    ING_STORE --> ING_LOG
    
    ETL_READ --> ETL_TRANS
    ETL_TRANS --> ETL_VAL
    ETL_VAL --> ETL_LOAD
    
    ML_PREP --> ML_FE
    ML_FE --> ML_TRAIN
    ML_TRAIN --> ML_EVAL
    ML_EVAL --> ML_REG
    
    PRED_API --> PRED_VAL
    PRED_VAL --> PRED_CLIENT
    PRED_CLIENT --> PRED_CACHE
    PRED_CLIENT --> PRED_LOG
```

### Component Details

| Component | Technology | Purpose | Scaling |
|-----------|------------|---------|---------|
| Ingestion Function | Cloud Functions (Python) | Fetch CQC data | Auto-scaling 0-100 |
| ETL Pipeline | Apache Beam on Dataflow | Transform and load data | Auto-scaling workers |
| ML Training | Vertex AI Pipelines | Train and evaluate models | Managed scaling |
| Prediction API | Cloud Functions (Python) | Serve predictions | Auto-scaling 0-1000 |
| Data Warehouse | BigQuery | Store and query data | Serverless |
| Model Registry | Vertex AI Model Registry | Version and manage models | Managed |
| Monitoring | Cloud Monitoring | System observability | Serverless |

## Security Architecture

### Security Layers
```mermaid
graph TB
    subgraph "External Layer"
        Internet[Internet]
        GCLB[Google Cloud<br/>Load Balancer]
    end
    
    subgraph "Application Layer"
        CF[Cloud Functions]
        IAM[Cloud IAM]
        SA[Service Accounts]
    end
    
    subgraph "Data Layer"
        SM[Secret Manager]
        GCS[Cloud Storage<br/>Encryption at Rest]
        BQ[BigQuery<br/>Column Encryption]
    end
    
    subgraph "Network Layer"
        VPC[VPC Network]
        PSC[Private Service<br/>Connect]
        FW[Firewall Rules]
    end
    
    Internet -->|HTTPS| GCLB
    GCLB -->|Authenticated| CF
    CF -->|IAM Roles| IAM
    CF -->|Uses| SA
    SA -->|Access| SM
    SA -->|Access| GCS
    SA -->|Access| BQ
    CF -.->|Private| VPC
    VPC --> PSC
    VPC --> FW
```

### Security Controls

| Layer | Control | Implementation |
|-------|---------|----------------|
| Authentication | API Keys | Secret Manager |
| Authorization | IAM Roles | Least privilege |
| Encryption | At Rest | Default GCS/BQ encryption |
| Encryption | In Transit | TLS 1.2+ |
| Network | Firewall | Deny all, allow specific |
| Audit | Logging | Cloud Audit Logs |
| Secrets | Management | Secret Manager with rotation |

## Deployment Architecture

### CI/CD Pipeline
```mermaid
graph LR
    subgraph "Development"
        Dev[Developer]
        Git[GitHub]
    end
    
    subgraph "CI/CD"
        CB[Cloud Build]
        CR[Container Registry]
        TF[Terraform]
    end
    
    subgraph "Environments"
        DEV_ENV[Dev Environment]
        STAGING[Staging]
        PROD[Production]
    end
    
    Dev -->|Push| Git
    Git -->|Trigger| CB
    CB -->|Build| CR
    CB -->|Deploy Infra| TF
    CB -->|Deploy Dev| DEV_ENV
    CB -->|Deploy Staging| STAGING
    CB -->|Manual Approval| PROD
```

### Infrastructure as Code
```mermaid
graph TD
    subgraph "Terraform Modules"
        NET[Network Module]
        IAM_MOD[IAM Module]
        STORAGE[Storage Module]
        COMPUTE[Compute Module]
        DATA[Data Module]
        ML[ML Module]
    end
    
    subgraph "Resources Created"
        VPC_RES[VPC & Subnets]
        SA_RES[Service Accounts]
        BUCKET[GCS Buckets]
        CF_RES[Cloud Functions]
        BQ_RES[BigQuery Dataset]
        VAI_RES[Vertex AI Resources]
    end
    
    NET --> VPC_RES
    IAM_MOD --> SA_RES
    STORAGE --> BUCKET
    COMPUTE --> CF_RES
    DATA --> BQ_RES
    ML --> VAI_RES
```

## Database Schema

### BigQuery Schema Design
```mermaid
erDiagram
    PROVIDERS ||--o{ LOCATIONS : has
    LOCATIONS ||--o{ ML_FEATURES : generates
    ML_FEATURES ||--o{ PREDICTIONS : used_for
    
    PROVIDERS {
        string provider_id PK
        string name
        string type
        string main_service
        timestamp created_date
        timestamp last_updated
    }
    
    LOCATIONS {
        string location_id PK
        string provider_id FK
        string name
        string type
        string region
        array regulated_activities
        float latitude
        float longitude
        timestamp last_inspection_date
        string overall_rating
        timestamp last_updated
    }
    
    ML_FEATURES {
        string location_id FK
        string provider_id FK
        int days_since_inspection
        int num_activities
        float historical_rating_avg
        int provider_location_count
        string region_encoded
        string type_encoded
        bool is_inherited_rating
        timestamp feature_timestamp
    }
    
    PREDICTIONS {
        string prediction_id PK
        string location_id FK
        timestamp prediction_timestamp
        string predicted_rating
        float confidence_score
        string model_version
        json feature_values
        string actual_rating
    }
```

### Table Partitioning Strategy
- **locations**: Partitioned by `last_updated` (daily)
- **ml_features**: Partitioned by `feature_timestamp` (daily)
- **predictions**: Partitioned by `prediction_timestamp` (daily)
- **providers**: Partitioned by `last_updated` (monthly)

## API Architecture

### REST API Design
```mermaid
graph TD
    subgraph "API Gateway"
        EP1[/predict]
        EP2[/batch_predict]
        EP3[/health]
        EP4[/metrics]
    end
    
    subgraph "Request Processing"
        VAL[Input Validation]
        AUTH[Authentication]
        RL[Rate Limiting]
        CACHE[Cache Layer]
    end
    
    subgraph "Business Logic"
        PRED[Prediction Service]
        BATCH[Batch Service]
        MON[Monitoring Service]
    end
    
    EP1 --> VAL
    EP2 --> VAL
    EP3 --> MON
    EP4 --> MON
    
    VAL --> AUTH
    AUTH --> RL
    RL --> CACHE
    CACHE --> PRED
    CACHE --> BATCH
```

### API Endpoints

| Endpoint | Method | Purpose | Rate Limit |
|----------|--------|---------|------------|
| `/predict` | POST | Single prediction | 100/min |
| `/batch_predict` | POST | Batch predictions | 10/min |
| `/health` | GET | Health check | 1000/min |
| `/metrics` | GET | Service metrics | 100/min |

### Request/Response Format

**Prediction Request:**
```json
{
  "instances": [{
    "location_id": "1-12345",
    "provider_id": "1-67890",
    "days_since_last_inspection": 180,
    "num_regulated_activities": 5,
    "region": "London",
    "type": "Residential social care",
    "historical_rating_avg": 3.5
  }]
}
```

**Prediction Response:**
```json
{
  "predictions": [{
    "location_id": "1-12345",
    "predicted_rating": "Good",
    "confidence": 0.85,
    "rating_probabilities": {
      "Outstanding": 0.10,
      "Good": 0.85,
      "Requires improvement": 0.04,
      "Inadequate": 0.01
    },
    "model_version": "cqc-xgboost-v2.1"
  }]
}
```

## ML Pipeline Architecture

### Training Pipeline
```mermaid
graph TD
    subgraph "Data Preparation"
        DS[Data Source<br/>BigQuery]
        FE[Feature<br/>Engineering]
        SPLIT[Train/Val/Test<br/>Split]
    end
    
    subgraph "Model Training"
        XGB[XGBoost<br/>Training]
        LGBM[LightGBM<br/>Training]
        AUTOML[AutoML<br/>Training]
    end
    
    subgraph "Evaluation"
        EVAL[Model<br/>Evaluation]
        COMP[Model<br/>Comparison]
        SEL[Model<br/>Selection]
    end
    
    subgraph "Deployment"
        REG[Model<br/>Registry]
        END[Endpoint<br/>Deployment]
        MON[Model<br/>Monitoring]
    end
    
    DS --> FE
    FE --> SPLIT
    SPLIT --> XGB
    SPLIT --> LGBM
    SPLIT --> AUTOML
    XGB --> EVAL
    LGBM --> EVAL
    AUTOML --> EVAL
    EVAL --> COMP
    COMP --> SEL
    SEL --> REG
    REG --> END
    END --> MON
```

### Feature Engineering Pipeline
```mermaid
graph LR
    subgraph "Raw Features"
        RF1[Location Data]
        RF2[Provider Data]
        RF3[Inspection History]
    end
    
    subgraph "Derived Features"
        DF1[Time-based<br/>Features]
        DF2[Aggregated<br/>Features]
        DF3[Encoded<br/>Features]
    end
    
    subgraph "Feature Store"
        FS[(BigQuery<br/>ML Features)]
    end
    
    RF1 --> DF1
    RF2 --> DF2
    RF3 --> DF1
    RF3 --> DF2
    RF1 --> DF3
    RF2 --> DF3
    DF1 --> FS
    DF2 --> FS
    DF3 --> FS
```

### Model Monitoring Architecture
```mermaid
graph TD
    subgraph "Monitoring Inputs"
        PRED[Predictions]
        ACTUAL[Actual Ratings]
        FEAT[Feature Distributions]
    end
    
    subgraph "Monitoring Metrics"
        ACC[Accuracy Metrics]
        DRIFT[Data Drift]
        PERF[Performance Metrics]
    end
    
    subgraph "Alerting"
        ALERT[Alert Manager]
        DASH[Dashboards]
        RETRAIN[Retrain Trigger]
    end
    
    PRED --> ACC
    ACTUAL --> ACC
    FEAT --> DRIFT
    PRED --> PERF
    
    ACC --> ALERT
    DRIFT --> ALERT
    PERF --> ALERT
    
    ALERT --> DASH
    ALERT --> RETRAIN
```

## Performance Architecture

### Caching Strategy
```mermaid
graph LR
    subgraph "Request Flow"
        REQ[Request]
        CACHE{Cache Hit?}
        PRED[Prediction Service]
        RESP[Response]
    end
    
    subgraph "Cache Layers"
        L1[Cloud Function<br/>Memory Cache<br/>TTL: 5 min]
        L2[Redis Cache<br/>TTL: 1 hour]
        L3[BigQuery<br/>Results Cache<br/>TTL: 24 hours]
    end
    
    REQ --> CACHE
    CACHE -->|Yes| L1
    CACHE -->|No| PRED
    L1 -->|Miss| L2
    L2 -->|Miss| L3
    L3 -->|Miss| PRED
    PRED --> RESP
    L1 --> RESP
    L2 --> RESP
    L3 --> RESP
```

### Scaling Architecture
```mermaid
graph TD
    subgraph "Auto-scaling Components"
        CF_AS[Cloud Functions<br/>0-1000 instances]
        DF_AS[Dataflow<br/>2-100 workers]
        VAI_AS[Vertex AI Endpoints<br/>2-50 replicas]
    end
    
    subgraph "Scaling Triggers"
        CPU[CPU > 80%]
        MEM[Memory > 85%]
        LAT[Latency > 1s]
        QUEUE[Queue Depth > 100]
    end
    
    subgraph "Scaling Actions"
        SCALE_UP[Scale Up]
        SCALE_DOWN[Scale Down]
        ALERT_OPS[Alert Ops Team]
    end
    
    CPU --> SCALE_UP
    MEM --> SCALE_UP
    LAT --> SCALE_UP
    QUEUE --> SCALE_UP
    
    CPU --> CF_AS
    MEM --> CF_AS
    LAT --> VAI_AS
    QUEUE --> DF_AS
    
    CF_AS --> SCALE_DOWN
    DF_AS --> SCALE_DOWN
    VAI_AS --> SCALE_DOWN
    
    SCALE_UP --> ALERT_OPS
```

## Cost Optimization Architecture

### Resource Lifecycle
```mermaid
graph LR
    subgraph "Storage Lifecycle"
        HOT[Hot Storage<br/>0-30 days]
        COOL[Cool Storage<br/>31-90 days]
        ARCH[Archive<br/>91+ days]
        DEL[Delete<br/>365+ days]
    end
    
    subgraph "Compute Optimization"
        SPOT[Spot/Preemptible<br/>Training]
        SCHED[Scheduled<br/>Scaling]
        IDLE[Idle Detection]
    end
    
    HOT --> COOL
    COOL --> ARCH
    ARCH --> DEL
    
    SPOT --> SCHED
    SCHED --> IDLE
```

## Disaster Recovery Architecture

### Multi-Region Setup
```mermaid
graph TB
    subgraph "Primary Region (europe-west2)"
        P_CF[Cloud Functions]
        P_BQ[BigQuery]
        P_GCS[Cloud Storage]
        P_VAI[Vertex AI]
    end
    
    subgraph "DR Region (europe-west1)"
        DR_CF[Cloud Functions<br/>Standby]
        DR_BQ[BigQuery<br/>Replica]
        DR_GCS[Cloud Storage<br/>Mirror]
        DR_VAI[Vertex AI<br/>Backup]
    end
    
    subgraph "Replication"
        SYNC[Real-time Sync]
        BACKUP[Daily Backup]
    end
    
    P_BQ -.->|Cross-region<br/>replication| DR_BQ
    P_GCS -.->|Multi-region<br/>bucket| DR_GCS
    P_VAI -.->|Model<br/>backup| DR_VAI
    P_CF -.->|Config<br/>sync| DR_CF
    
    P_BQ --> SYNC
    P_GCS --> SYNC
    P_VAI --> BACKUP
    SYNC --> DR_BQ
    SYNC --> DR_GCS
    BACKUP --> DR_VAI
```

## Summary

This architecture provides:
- **Scalability**: Auto-scaling across all components
- **Reliability**: Multi-region DR, health checks, monitoring
- **Security**: Defense in depth, encryption, IAM
- **Performance**: Caching, CDN, optimized queries
- **Cost Efficiency**: Lifecycle policies, spot instances, serverless

The system is designed to handle:
- 1M+ predictions per day
- 100GB+ data ingestion weekly
- Sub-second prediction latency
- 99.9% availability SLA