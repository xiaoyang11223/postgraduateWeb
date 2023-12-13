import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# 页面内容设置
# 页面名称
st.set_page_config(page_title="Readmission", layout="wide")
# 标题
st.title('The machine-learning based model to predict Readmission')
# 文本
st.write('This is a web app to predict the prob of Readmission based on\
         several features that you can see in the sidebar. Please adjust the\
         value of each feature. After that, click on the Predict button at the bottom to\
         see the prediction.')

st.markdown('## Input Data:')
# 隐藏底部水印
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            <![]()yle>
            """

st.markdown(hide_st_style, unsafe_allow_html=True)

def option_name(x):
    if x == 0:
        return "no"
    if x == 1:
        return "yes"
@st.cache
def predict_quality(model, df):
    y_pred = model.predict(df, prediction_type='Probability')
    return y_pred[:, 1]

# 导入模型
model = joblib.load('catb-rfecatb.pkl')##导入相应的模型
st.sidebar.title("Features")

# 设置各项特征的输入范围和选项
LVEF = st.sidebar.number_input(label='LVEF', min_value=10.0,
                                  max_value=90.0,
                                  value=22.0,
                                  step=1.0)
LAD = st.sidebar.number_input(label='LAD', min_value=10.0,
                                  max_value=90.0,
                                  value=15.0,
                                  step=1.0)
Age = st.sidebar.number_input(label='Age', min_value=65.0,
                                  max_value=100.0,
                                  value=65.0,
                                  step=1.0)
LOS = st.sidebar.number_input(label='LOS', min_value=1.0,
                                  max_value=90.0,
                                  value=1.0,
                                  step=1.0)
IVST = st.sidebar.number_input(label='IVST', min_value=5.0,
                                  max_value=20.0,
                                  value=5.0,
                                  step=0.1)

NTproBNP = st.sidebar.number_input(label='NT.proBNP', min_value=5.0,
                                  max_value=67000.0,
                                  value=70.0,
                                  step=100.0
                                   )
Cl = st.sidebar.number_input(label='Cl', min_value=65.0,
                                  max_value=125.0,
                                  value=65.0,
                                  step=1.0
                                   )

APTT = st.sidebar.number_input(label='APTT', min_value=15.0,
                                  max_value=100.0,
                                  value=15.0,
                                  step=1.0
                                   )

LYM = st.sidebar.number_input(label='LYM', min_value=1.0,
                                  max_value=70.0,
                                  value=1.0,
                                  step=0.1
                                   )
ALT = st.sidebar.number_input(label='ALT', min_value=1.0,
                                  max_value=200.0,
                                  value=1.0,
                                  step=0.1
                                   )

Creatinine = st.sidebar.number_input(label='Creatinine', min_value=20.0,
                                  max_value=200.0,
                                  value=20.0,
                                  step=0.1
                                   )
GGT = st.sidebar.number_input(label='GGT', min_value=3.0,
                                  max_value=150.0,
                                  value=3.0,
                                  step=0.1
                                   )
TT= st.sidebar.number_input(label='TT', min_value=10.0,
                                  max_value=60.0,
                                  value=10.0,
                                  step=0.1
                                   )
LDLcholesterol= st.sidebar.number_input(label='LDL.cholesterol', min_value=0.31,
                                  max_value=8.0,
                                  value=0.31,
                                  step=0.01
                                   )

Albumin= st.sidebar.number_input(label='Albumin', min_value=17.0,
                                  max_value=50.0,
                                  value=17.0,
                                  step=1.0
                                   )

LEU= st.sidebar.number_input(label='LEU', min_value=1.0,
                                  max_value=30.0,
                                  value=1.0,
                                  step=0.01
                                   )

PT= st.sidebar.number_input(label='PT', min_value=8.8,
                                  max_value=60.0,
                                  value=8.8,
                                  step=0.1
                                   )


RBC= st.sidebar.number_input(label='RBC', min_value=1.1,
                                  max_value=8.0,
                                  value=1.1,
                                  step=0.01
                                   )

AST = st.sidebar.number_input(label='AST', min_value=4.4,
                                  max_value=100.0,
                                  value=4.4,
                                  step=0.1
                                   )
SBP = st.sidebar.number_input(label='SBP', min_value=50.0,
                                  max_value=200.0,
                                  value=50.0,
                                  step=0.1
                                   )
Neu = st.sidebar.number_input(label='Neu', min_value=12.4,
                                  max_value=98.0,
                                  value=12.4,
                                  step=0.1
                                   )

DBP = st.sidebar.number_input(label='DBP', min_value=24.0,
                                  max_value=150.0,
                                  value=24.0,
                                  step=1
                                   )

Uricacid = st.sidebar.number_input(label='Uric.acid', min_value=52.0,
                                  max_value=502.0,
                                  value=52.0,
                                  step=0.1
                                   )

PLT = st.sidebar.number_input(label='PLT', min_value=1.0,
                                  max_value=220.0,
                                  value=1.0,
                                  step=1
                                   )




features = {'LVEF': LVEF, 'LAD': LAD,
            'Age': Age, 'LOS': LOS,
            'IVST': IVST, 'NT.proBNP': NTproBNP,
            'Cl': Cl, 'APTT': APTT,
            'LYM': LYM, 'ALT': ALT,
            'Creatinine': Creatinine, 'GGT': GGT,
            'TT': TT, 'LDL.cholesterol': LDLcholesterol,
            'Albumin': Albumin, 'LEU': LEU,
            'PT': PT, 'RBC': RBC,
            'AST': AST, 'SBP': SBP,
            'Neu': Neu,'DBP': DBP, 'Uric.acid': Uricacid,
            'PLT': PLT
            }

features_df = pd.DataFrame([features])
#显示输入的特征
st.table(features_df)

#显示预测结果与shap解释图
if st.button('Predict'):
    prediction = predict_quality(model, features_df)
    st.write("the probability of Readmission:")
    st.success(round(prediction[0], 4))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features_df)
    shap.force_plot(explainer.expected_value, shap_values[0], features_df, matplotlib=True, show=False)
    plt.subplots_adjust(top=0.67,
                        bottom=0.0,
                        left=0.1,
                        right=0.9,
                        hspace=0.2,
                        wspace=0.2)
    plt.savefig('test_shap.png')
    st.image('test_shap.png', caption='Individual prediction explaination', use_column_width=True)
