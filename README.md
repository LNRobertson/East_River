![image](https://github.com/user-attachments/assets/cbe5e2f7-2fbc-48f0-999c-fbdfacc7e890)
 

## East River General Load Prediction & Peak Flagging POC 

Contributors:
- [Lindsey Robertson](https://linkedin.com/in/lnrobertson)
- [Aman Singh](https://www.linkedin.com/in/iamasr999)

Objectives 

This project develops an AI-powered load forecasting model for East River General that aims to: 

- ✅Predict Estimated Online Load (MW) for the next 24–72 hours using historical SCADA and regional weather data. 
- ✅Flag potential peak demand events before they exceed the Control Threshold (MW)
- ✅Incorporate periodic or regularly updated weather data sources to improve forecast accuracy. 


Business Goal: Reduce manual load control interventions and optimize grid stability while preventing costly peak demand charges—even with limited resources. 

- Find progress and approach [slide deck here](https://docs.google.com/presentation/d/1pJottihB1ASlbE_PxFcr8F8a7vs-HiHBUzuatYwBiy0/edit?usp=sharing)

Problem Statement 

Operators manually control load (Load_Control_MW) when Estimated_OnLine_Load_MW approaches the Control Threshold (Control_Threshold_MW). 

- ❌ Existing forecasting models are not accurate enough, leading to unexpected peaks and last-minute interventions. 
- ❌ Traditional ML models (like XGBoost) did not capture long-range dependencies well. 
- ❌ Weather significantly affects load, but previous methods struggled to incorporate it effectively. 


🛠 Tech Stack 


Programming: Python 

- ML Libraries: scikit-learn, XGBoost/LightGBM, PyTorch for LSTM 


Data Integration/Ingestion:  


Deployment:  


API/Dashboard: 


✅ Solution 

Proposed Approach for the POC: 

Data Integration & Feature Engineering: 

- Use historical SCADA data alongside historical weather data provided by East River to understand past patterns and train our models effectively.  

- Fill in missing precipitation weather data 

- Supplement with free, high-resolution weather forecasts (from Open-Meteo or similar) to predict future weather conditions. 

- Blend in alternate data like Active Water Heaters, ground moisture, public holiday and event data to flag days that could show atypical load patterns. 

- Generate lagged weather variables and calendar-based features to enrich the model inputs. 


Modeling: 

- Start with a baseline, lightweight open-source model (e.g. XGBoost/LightGBM) to quickly establish forecasting accuracy. Compare with previous solution.  

- Optionally, experiment with a simple LSTM or shallow neural network to capture sequential patterns if time and resources allow. 

- Incorporate uncertainty estimates for improved peak flagging confidence. 


Operational Integration: 

Generate a forecast for Estimated Load Control MW and flag peaks based on operator input or day and location. 

Automate forecasts on a schedule? 

 

Blend public holiday and local event data to capture non-weather influences on load spikes. 

 

 

🚀 Installation & Setup 

 

Dependencies 

 

📊 Data Sources  

1️⃣ SCADA Data (Load & Control Information): 

Source: East River General SCADA System 

Key Variables: 

OnLine_Load_MW: Actual measured online load. 

Load_Control_MW: Load manually controlled by operators. 

Estimated_OnLine_Load_MW: Prediction target variable. 

Control_Threshold_MW: Threshold above which manual intervention is triggered. 

Note: Ingest both historical and live SCADA data to ensure real-time forecasting and model validation. 

2️⃣ Weather Data: 

Historical Weather Data: 

Source: Provided by East River 

Usage: Used for training, validating, and feature engineering (e.g., lagged weather effects, seasonal trends) to understand how past weather events impacted load. 

Key variables like; temperature, humidity, precipitation. 

Forecast Weather Data: 

Source: Free weather forecast APIs (e.g., Open-Mateo API Documentation, visual crossing) 

Key Features: temperature, humidity, precipitation, etc., used to predict future conditions. 

 

3️⃣ External Factors – Public Holidays & Events: 

Source: Public APIs (e.g., HolidayAPI) or available CSV calendars. 

Key Features: 

Binary/categorical flags to indicate public holidays or significant local events. 

These flags help detect usage spikes that are not solely weather-dependent. 

 

🏗 Model Architecture 

1️⃣ Baseline Models: 

XGBoost / LightGBM: 

Serve as the starting point for forecasting performance, providing quick, efficient training and prediction. 

2️⃣ Advanced Models Under Consideration: 

Simple LSTM/GRU Models: 

To capture sequential dependencies if additional accuracy is required. 

Temporal Fusion Transformer (TFT): 

While TFT is a promising model for capturing long-range dependencies and dynamic feature importance, it is somewhat ambitious given our current resource constraints and experience. We note TFT as a potential future enhancement once our baseline model is validated and we are ready to invest in more advanced methodologies. 

Future Exploration – Graph Neural Networks: 

May be considered if grid-wide spatial dependencies need to be modeled. 

 

Why This Approach? 
✔ Leverages cost-effective, open-source tools and free data sources. 
✔ Integrates dynamic weather data with non-weather features (holidays/events) to improve prediction robustness. 
✔ Provides uncertainty estimates to enhance peak flagging confidence, meeting both operational and stakeholder needs. 

✔ Although advanced models like the Temporal Fusion Transformer (TFT) show significant promise, adopting them at this stage would add considerable complexity. We choose to begin with proven, lightweight models like XGBoost/LightGBM, which are well-suited to our current resources and timeline—while keeping advanced options like TFT in our roadmap for future enhancements. These future enhancements, including sophisticated sequential modeling, dynamic attention mechanisms, and improved integration of multi-modal data (such as uncertainty quantification and richer weather/event features), are expected to significantly outperform previous solutions. 

 

Preprocessing 

1️⃣ SCADA data is merged into one dataset. Historical weather data will be parsed to convert timestamps into datetime objects, handle data types, conduct initial EDA and transform datetime features into half our intervals to align the weather data with SCADA observations. Forward fill missing values in weather data. Supplement missing weather information with alternate historical weather data. Add additional features from alternate data sources.  
 
2️⃣ Merge into one data frame on datetime. Feature engineering:  

3️⃣  

 

 

Research 

Our approach builds on insights from recent research on AI forecasting and dynamic data integration in grid demand management. 

 

Research Paper 1git s 

Research Paper 2 

Research Paper 3 

 

🔍 Key Open Questions 

How can we best quantify uncertainty to enhance the reliability of peak flagging? 

What is the incremental value of including public holiday/event data versus weather data alone? 

At what point should we consider adding more advanced models (e.g., LSTM or Graph Neural Networks) given resource constraints? 

 

🗺️ Possible Next Steps & Roadmap 

Initial Validation: 

Run the POC with historical data to benchmark forecast accuracy and assess the impact of blended data sources (weather + holiday/event data). 

Stakeholder Feedback: 

Present preliminary results via the dashboard and gather feedback on forecast performance and the utility of peak flagging. 

Iterative Refinement: 

Fine-tune the baseline model (e.g., XGBoost/LightGBM) based on feedback. 

Evaluate whether additional data sources (e.g., energy market prices, renewable injections) further improve accuracy. 

Exploration of Advanced Methods: 

Hybrid Approaches: Investigate combining physics-based forecasts with our ML model (e.g., using a bias-correction layer) to further boost accuracy and robustness. 

Temporal Fusion Transformer (TFT): Evaluate TFT as an advanced alternative to capture long-range dependencies and dynamic feature importance, especially if our initial models indicate room for improvement. 

Scaling & Production: 

Transition from the POC to a production-ready system with scheduled updates and real-time API integration once performance and stability are established. 

Future Consideration: 

Should resources allow, consider exploring cutting-edge DeepMind models (e.g., GraphCast or WeatherNext) as a potential avenue for further performance enhancements. This option can be revisited once the baseline approach is validated, and additional investment is justified. 

Graph Neural Networks: 

May be considered if grid-wide spatial dependencies need to be modeled. 

Energy Market Prices and Renewable Energy Injection data can be integrated later for further refinement. 

Quantifying Savings: 
Comparing the costs associated with manual load control interventions, peak demand penalties, and energy wastage in historical operations against our model's forecasts and automated interventions. This comparison, using our historical SCADA data and operational cost metrics (if available), will help demonstrate the economic value of improved forecasting. 

Create a schedule for retraining and updating the model or create continuous improvement pipelines.  

 

📜 License 

[MIT License] 

 

📬 Contact 

For questions or collaboration inquiries, please contact: 
[Contact Information] 

 
