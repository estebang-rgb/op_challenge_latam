# LATAM Software Engineer (ML & LLMs) Challenge - My Solution

This was a challenge that really let me dive deep into operationalizing a machine learning model. I went from a messy Jupyter notebook to a production-ready system with APIs, cloud deployment, and all the bells and whistles. Let me walk you through what I built and why I made the decisions I did.

## The Challenge

The task was to take a data scientist's exploratory notebook for flight delay prediction at SCL airport and turn it into something production-ready. Four main pieces needed to be built:

1. **Clean up the model code** - Turn that notebook spaghetti into proper Python classes
2. **Build a REST API** - Make predictions accessible via HTTP calls
3. **Deploy to the cloud** - Get it running on GCP so it's actually usable
4. **Set up CI/CD** - Make sure everything stays working as we iterate

## How I Architected This

I wanted something simple but production-ready. Here's what the final system looks like:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI App   │    │   DelayModel    │    │   XGBoost       │
│                 │    │   - preprocess  │    │   Classifier    │
│ - /predict      │◄──►│   - fit         │◄──►│   (balanced)    │
│ - /health       │    │   - predict     │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌─────────────────────┐
                    │   Docker Container  │
                    │   Google Cloud Run  │
                    │   Load Balanced     │
                    └─────────────────────┘
```

FastAPI handles the web requests, calls into my DelayModel class for preprocessing and prediction, which uses XGBoost under the hood. Everything gets packaged in a Docker container and deployed to Google Cloud Run for auto-scaling. Clean and simple!

## Part I: The Model - My Biggest Learning Moment

This was where I really dug into the data science work. Let me tell you about what I found.

### Why I Chose XGBoost + Class Balancing

I started by actually running the notebook experiments and checking the real performance numbers. What I discovered completely changed my perspective on this problem. The data scientist had left a question at the end wondering which model to pick, and I had to figure it out.

The key insight? **Class balancing isn't optional - it's absolutely critical.** Without it, the models are basically useless for delay prediction.

### The Shocking Performance Results

I ran all the experiments from the notebook and here's what actually happened (these are the real numbers, not made up):

| Model Configuration | Class 1 Recall | Class 1 F1-Score | Class 0 Recall | Class 0 F1-Score |
|---------------------|----------------|------------------|----------------|------------------|
| XGBoost (all features) | 0.00 | 0.00 | 1.00 | 0.90 |
| XGBoost (top 10 + balance) | **0.69** | **0.37** | 0.52 | 0.66 |
| XGBoost (top 10, no balance) | 0.01 | 0.01 | 1.00 | 0.90 |
| Logistic Regression (all features) | 0.03 | 0.06 | 0.99 | 0.90 |
| Logistic Regression (top 10 + balance) | **0.69** | **0.36** | 0.52 | 0.65 |
| Logistic Regression (top 10, no balance) | 0.01 | 0.03 | 1.00 | 0.90 |

### What Blew My Mind

1. **The Class Imbalance Disaster**: Only 18% of flights are delayed, but without balancing, models predict ZERO delays correctly. That's not just bad - it's dangerous for a prediction system!

2. **Balancing is a Game Changer**: Add class balancing and suddenly we get 69% recall. The difference between "useless" and "actually useful" is one parameter.

3. **Less is More**: The top 10 features work just as well as all 25+, but train faster and are easier to manage.

4. **XGBoost vs Logistic Regression**: When balanced, they're basically identical! The DS was right to wonder which to pick.

5. **Finally Useful for Business**: 69% delay detection means the airport can actually prepare for problems instead of reacting to them.

### So I Went With XGBoost

Even though XGBoost and Logistic Regression perform basically identically here, I chose XGBoost because:

- **It's more robust** - handles weird data better than Logistic Regression
- **Missing data? No problem** - XGBoost deals with it gracefully
- **Industry standard** - everyone uses it for tabular data in production
- **Future-proof** - if we add more features later, XGBoost will handle it better

**Answering the DS's Question**: "With this, the model to be productive must be the one that is trained with the top 10 features and class balancing, but which one?"

Between XGBoost and Logistic Regression (both with balancing), I'd pick XGBoost. They perform the same here, but XGBoost is more reliable in the real world.


#### Production-Ready Code
I made the code much more robust:
- Added type hints everywhere
- Comprehensive error handling with sensible defaults
- Clean separation between feature engineering and model logic
- Proper documentation for future maintainers

## Part II: The API - Making It Usable

Now that I had a working model, I needed to expose it as a web service. FastAPI was the obvious choice - it's fast, has great type checking, and generates OpenAPI docs automatically.

### Simple, Clean Endpoints

I kept it simple with just two endpoints:

**GET /health** - For load balancers and monitoring
```json
{"status": "OK"}
```

**POST /predict** - The main prediction endpoint
```json
// Request
{
  "flights": [{
    "OPERA": "Aerolineas Argentinas",
    "TIPOVUELO": "N",
    "MES": 3
  }]
}

// Response
{"predict": [0]}  // 0 = on time, 1 = delayed
```

### Design Decisions

- **Batch predictions** - Send multiple flights in one request for efficiency
- **Strict validation** - Month must be 1-12, flight type must be "I" or "N"
- **Graceful handling** - Unknown airlines get processed (the model handles them via feature engineering)
- **Proper HTTP codes** - 400 for bad requests, 500 for server errors

The model loads once at startup and gets reused for all predictions. Smart caching keeps things fast!

## Part III: Going Live - Cloud Deployment

Time to get this thing running in the cloud! I chose Google Cloud Platform because it's what the challenge recommended.


### Container Setup

I created a simple Dockerfile that:
- Uses Python 3.11 slim for a smaller image
- Installs only the XGBoost dependencies we need
- Copies just the essential code
- Runs as non-root user for security
- Includes health checks for monitoring

Boom! API is live and auto-scaling. The airport team can now call it from anywhere.


## Part IV: CI/CD - Keeping Things Working

I set up GitHub Actions to automate everything. No more "works on my machine" excuses!

### What the CI Pipeline Does

Every time someone pushes code or opens a PR:
- **Runs all tests** - model tests, API tests, everything
- **Checks code quality** - linting with flake8, formatting with black
- **Builds the package** - makes sure everything compiles
- **Coverage reports** - gotta keep that test coverage 

### CD Makes Deployment Effortless

When code gets merged to main:
- **Builds the Docker image** and pushes to Google Container Registry
- **Deploys to Cloud Run** with zero-downtime updates
- **Runs smoke tests** to make sure the API actually works
- **Sends notifications** so the team knows deployment status

### Git Flow Setup

I organized the repo with proper branching:
```
main (production deployments)
├── develop (integration branch)
│   ├── feature/model-cleanup
│   ├── feature/api-endpoints
│   ├── feature/cloud-deploy
│   └── feature/ci-pipeline
```

### Quality Gates

- Tests must pass a
- Code must be properly formatted
- No security vulnerabilities in dependencies
- API must respond to health checks after deployment

This setup means the airport team gets reliable, tested code every time. No more deployment anxiety!


### What I Test

**Model Tests**: Making sure feature preprocessing works correctly, the model trains properly, and predictions make sense.

**API Tests**: Verifying the endpoints work, input validation catches bad data, and error messages are helpful.

**Stress Tests**: Using Locust to simulate real traffic and make sure the system can handle load.


### Performance That Matters

**The Model:**
- **69% delay detection** - This is what the airport actually cares about
- **37% F1 balance** - Reasonable precision/recall tradeoff
- **Fast training** - Top 10 features instead of 25+ means quicker iteration


## Wrapping Up - 

The airport team now has a working delay prediction API that can actually help them prepare for problems instead of just reacting to them. From messy notebook code to cloud-deployed API with tests and CI/CD - that's the journey of operationalizing ML.

The system is live, tested, and ready to help SCL airport run more smoothly. Mission accomplished