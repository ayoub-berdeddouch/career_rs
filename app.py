import flask
import pandas as pd
import numpy as np

from tensorflow.keras.models import load_model

#from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer

import json
import plotly
import plotly.express as px

import lime
import lime.lime_tabular

# QuantileTransformer to Normal distrubution.
qt = QuantileTransformer(output_distribution='normal')


# Loading Model
model = load_model("models/model_career_RS_10classes.h5")


class_names = [
                    'BUSINESS',
                    'SPORTS AND PHYSICAL TRAIN',
                    'ENGINEERING',
                    'HUMANITIES AND SOCIAL SCIENCE',
                    'MATH AND PHYSICAL SCIENCES',
                    'NUTRITION AND DIETETICS',
                    'HEALTH & MEDICINE',
                    'ARTS AND DESIGN',
                    'BIOLOGICAL SCIENCE',
                    'PLASTIC ARTS, VISUAL ARTS',
                ]



 # for LIME 

with open('models/RS_X_training.bin','rb') as f:
    X_train = np.load(f)
    f.close()

# f = file("models/RS_X_training.bin","rb")
# X_train = np.load(f)
# f.close()



# routes 

app = flask.Flask(__name__, template_folder='templates')
@app.route('/')
def main():
    return (flask.render_template('index.html'))

@app.route('/report1')
def report():
    return (flask.render_template('report1_major.html'))

@app.route('/report2')
def jointreport():
    return (flask.render_template('report2_minor.html'))


@app.route("/career_rs", methods=['GET', 'POST'])
def Career_Recommendation():
    
    if flask.request.method == 'GET':
        return (flask.render_template('Career_RS.html'))
    
    if flask.request.method =='POST':
        
        #get Scores input
        
        score_language       =  int(flask.request.form['language'])
        score_mathematics    =  int(flask.request.form['mathematic'])
        score_biology        =  int(flask.request.form['biology'])
        score_chemistry      =  int(flask.request.form['chemistry'])
        score_physics        =  int(flask.request.form['physics'])
        score_social_science =  int(flask.request.form['social_science'])
        score_philosophy     =  int(flask.request.form['philosophy'])
        score_english        =  int(flask.request.form['english'])

        
        #create original output dict
        output_dict= dict()
        output_dict['score_language'] = score_language
        output_dict['score_mathematics'] = score_mathematics
        output_dict['score_biology'] = score_biology
        output_dict['score_chemistry']=score_chemistry
        output_dict['score_physics'] = score_physics
        output_dict['score_social_science'] = score_social_science
        output_dict['score_philosophy'] = score_philosophy
        output_dict['score_english'] = score_english
        
        print(output_dict)
        

        X = [score_language,score_mathematics,score_biology,score_chemistry,score_physics,score_social_science,score_philosophy,score_english]
        
        print('------this is array data to before transformation-------')
        print('X = '+str(X))
        print('------------------------------------------')

        XX = qt.fit_transform(np.array(X).reshape(1,-1))
        print('------this is array data to predict after transformation-------')
        print('X = '+str(XX))
        print('------------------------------------------')
        
        
        probs = model.predict([XX])
        pred_class = np.argmax(probs)
        pred_class_name = class_names[pred_class]
        proba1 = int(probs[0][pred_class]*100)
        
        result = [pred_class_name,proba1]
        res = f'Top Career based on the Career_Recommendation System is {pred_class_name} with probability of {int(probs[0][pred_class]*100)}%'
        
        #################################
        # classes with last model of 13 Major Cores
        # classes = {0: 'BUSINESS',1: 'SPORTS AND PHYSICAL TRAIN',2: 'ENGINEERING',3: 'AGRONOMIC, LIVESTOCK ENGINEERING',
        #         4: 'HUMANITIES AND SOCIAL SCIENCE',5: 'MATH AND PHYSICAL SCIENCES',6: 'NUTRITION AND DIETETICS',
        #         7: 'HEALTH & MEDICINE',8: 'ARTS AND DESIGN',9: 'BIOLOGICAL SCIENCE',10: 'AGRICULTURAL, FOREST ENGINEERING',
        #         11: 'PLASTIC ARTS, VISUAL ARTS',12: 'PHISICS'}

        ############################
        # classes with latest model of 10 Major Cores
        classes = {
                     0: 'BUSINESS',
                     1: 'SPORTS AND PHYSICAL TRAIN',
                     2: 'ENGINEERING',
                     3: 'HUMANITIES AND SOCIAL SCIENCE',
                     4: 'MATH AND PHYSICAL SCIENCES',
                     5: 'NUTRITION AND DIETETICS',
                     6: 'HEALTH & MEDICINE',
                     7: 'ARTS AND DESIGN',
                     8: 'BIOLOGICAL SCIENCE',
                     9: 'PLASTIC ARTS, VISUAL ARTS',
                }
        ################################################
        class_labels = list(classes.values())
        probs_p = [np.round(x,6) for x in probs]
        fig = px.bar(x=probs_p, y=class_labels, color=  class_labels, orientation='h')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        fig.update_layout(xaxis_title="Probability", yaxis_title="Major Core", plot_bgcolor='rgb(255,255,255)')

        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        header="Career Recommendations based on our Engine"
        description = """
        These recommendations are based on your 8 Subjects Scores.
        """
        ##################################
        # Explain samples in XX
        X_explain = XX
        explainer = lime.lime_tabular.LimeTabularExplainer(training_data=X_train,
                                                           feature_names=['score_language_saber_11', 'score_mathematics_saber_11',
                                                                           'score_biology_saber_11', 'score_chemistry_saber_11',
                                                                           'score_physics_saber_11', 'score_social_science_saber_11',
                                                                           'score_philosophy_saber_11', 'score_english_saber_11'],
                                                           discretize_continuous=True,
                                                           class_names=['BUSINESS','SPORTS AND PHYSICAL TRAIN','ENGINEERING',
                                                                 'HUMANITIES AND SOCIAL SCIENCE','MATH AND PHYSICAL SCIENCES',
                                                                 'NUTRITION AND DIETETICS','HEALTH & MEDICINE','ARTS AND DESIGN',
                                                                 'BIOLOGICAL SCIENCE','PLASTIC ARTS, VISUAL ARTS'],
                                                           mode="classification",
                                                           verbose=True,
                                                           random_state=123)

        #Explaining first subject in test set using all 8 features
        exp = explainer.explain_instance(X_explain[0,:],
                                         model.predict, 
                                         num_features=8)
        exp = exp.as_html()
        #Plot local explanation
        #exp.as_pyplot_figure()
        #plt.tight_layout()
        #exp.show_in_notebook(show_table=True)


        ##################################

        return flask.render_template('Career_RS.html',
                                        original_input=output_dict, 
                                        result=result,
                                        graphJSON=graphJSON, 
                                        header=header,
                                        description=description,
                                        exp=exp,
                                    )


        
        
        
      
if __name__ == '__main__':
    app.run(debug=True)