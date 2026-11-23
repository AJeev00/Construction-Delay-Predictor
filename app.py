import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from pathlib import Path

st.set_page_config(page_title="SiteRisk | Delay Intelligence",page_icon="📈",layout="wide")
st.title("SiteRisk")
st.caption("Construction Delay Risk Intelligence | ML demonstration with simulated historical activities")
df=pd.read_csv(Path(__file__).parent/"data"/"activity_history.csv")
features=["activity_type","weather_disruption_days","material_delay_days","labour_shortfall_pct","design_change_count","dependency_delay_days","planned_duration_days"]
target="delayed"
cat=["activity_type"]; num=[x for x in features if x not in cat]
prep=ColumnTransformer([("cat",OneHotEncoder(handle_unknown="ignore"),cat),("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num)])
model=Pipeline([("prep",prep),("rf",RandomForestClassifier(n_estimators=250,max_depth=7,min_samples_leaf=3,class_weight="balanced",random_state=42))])
X=df[features]; y=df[target]
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
model.fit(Xtr,ytr); pred=model.predict(Xte); probs=model.predict_proba(Xte)[:,1]
a,b,c=st.columns(3)
a.metric("Holdout accuracy",f"{accuracy_score(yte,pred)*100:.1f}%")
b.metric("Delay precision",f"{precision_score(yte,pred):.2f}")
c.metric("Delay recall",f"{recall_score(yte,pred):.2f}")
st.subheader("Historical risk patterns")
col1,col2=st.columns(2)
with col1:
    st.plotly_chart(px.histogram(df,x="material_delay_days",color="delayed",barmode="overlay",nbins=15,title="Material delay days vs observed outcome"),use_container_width=True)
with col2:
    st.plotly_chart(px.box(df,x="delayed",y="labour_shortfall_pct",color="delayed",title="Labour shortfall distribution"),use_container_width=True)
st.subheader("Assess a planned activity")
with st.form("risk_form"):
    c1,c2,c3=st.columns(3)
    activity=c1.selectbox("Activity type",sorted(df.activity_type.unique()))
    weather=c2.slider("Expected weather disruption (days)",0,20,3)
    material=c3.slider("Material delay (days)",0,30,4)
    labour=c1.slider("Labour shortfall (%)",0,100,10)
    design=c2.slider("Expected design changes",0,10,1)
    dependency=c3.slider("Dependency delay (days)",0,30,2)
    duration=st.number_input("Planned activity duration (days)",min_value=1,max_value=365,value=30)
    submitted=st.form_submit_button("Estimate delay risk",type="primary")
if submitted:
    row=pd.DataFrame([dict(activity_type=activity,weather_disruption_days=weather,material_delay_days=material,labour_shortfall_pct=labour,design_change_count=design,dependency_delay_days=dependency,planned_duration_days=duration)])
    risk=float(model.predict_proba(row)[0,1])
    st.metric("Estimated delay probability",f"{risk*100:.1f}%")
    if risk>=.65: st.error("High risk: review procurement, labour deployment and predecessor readiness.")
    elif risk>=.35: st.warning("Moderate risk: monitor critical inputs and confirm mitigation owners.")
    else: st.success("Lower estimated risk: continue routine monitoring.")
    st.caption("Model output is a statistical estimate from synthetic training data, not a guaranteed forecast.")
st.subheader("Model evaluation")
cm=confusion_matrix(yte,pred)
st.plotly_chart(px.imshow(cm,text_auto=True,x=["Predicted on-time","Predicted delayed"],y=["Actual on-time","Actual delayed"],color_continuous_scale="Blues",title="Holdout confusion matrix"),use_container_width=True)
st.download_button("Download sample history",df.to_csv(index=False).encode(),"activity_history.csv","text/csv")
st.info("Synthetic data is used. A production model requires representative, validated project history, temporal validation, calibration and ongoing monitoring.")
