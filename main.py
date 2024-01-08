from flask import Flask, request;
from flask_cors import CORS, cross_origin;
from json import JSONEncoder;
import json;
import numpy
from decisionTree import decisionTree;
from linearRegression import linearRegression;
from neuralNetwork import neuralNetwork;
from randomForest import randomForest;
from ridgeRegression import ridgeRegression;
from svm import svm;

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


@app.route('/ml')
def mlResult():
    rangePercent = float(request.args.get('rangePercent', 0.1))

    dt = decisionTree(rangePercent);
    lR = linearRegression(rangePercent);
    nN = neuralNetwork(rangePercent);
    rF = randomForest(rangePercent);
    rR = ridgeRegression(rangePercent);
    sv = svm(rangePercent);

    Results = {};
    Results['decisionTree'] = dt['accuracy'];
    Results['linearRegression'] = lR['accuracy'];
    Results['neuralNetwork'] = nN['accuracy'];
    Results['randomForest'] = rF['accuracy'];
    Results['ridgeRegression'] = rR['accuracy'];
    Results['svm'] = sv['accuracy'];
    Results['rangePercent'] = rangePercent;

    return json.dumps(Results, cls=NumpyArrayEncoder)

@app.route('/mlPredict')
def mlPredict():
    D = float(request.args.get('D', 0.1))
    t = float(request.args.get('t', 0.1))
    L = float(request.args.get('L', 0.1))
    d = float(request.args.get('d', 0.1))
    YS = float(request.args.get('YS', 0.1))
    UTS = float(request.args.get('UTS', 0.1))
    Exp = float(request.args.get('Exp', 0.1))
    B31G = float(request.args.get('B31G', 0.1))
    MB31G = float(request.args.get('MB31G', 0.1))
    DNV = float(request.args.get('DNV', 0.1))
    Battelle = float(request.args.get('Battelle', 0.1))
    Shell = float(request.args.get('Shell', 0.1))
    Netto = float(request.args.get('Netto', 0.1))
    School = float(request.args.get('School', 0.1))
    Population = float(request.args.get('Population', 0.1))
    Water = float(request.args.get('Water', 0.1))
    rangePercent = float(request.args.get('rangePercent', 0.3))
    Result = float(request.args.get('Result', 0))

    dt = decisionTree(rangePercent);
    lR = linearRegression(rangePercent);
    # nN = neuralNetwork(rangePercent);
    rF = randomForest(rangePercent);
    # rR = ridgeRegression(rangePercent);
    sv = svm(rangePercent);

    # print(D, t, L, d, YS, UTS, Exp, B31G, MB31G, DNV, Battelle ,Shell, Netto, School, Population, Water);
    # 324.0 9.8 500.0 7.0 452.0 542.0 12.0 8.7 13.8 11.3 10.0 7.7 2.5 114.0 0.0 270.0
    
    Results = {};
    Results['decisionTree'] = {};
    Results['decisionTree']['accuracy'] = dt['accuracy'];
    Results['decisionTree']['mse'] = dt['mse'];
    Results['decisionTree']['predict'] = dt['machine'].predict([[D, t, L, d, YS, UTS, Exp, B31G, MB31G, DNV, Battelle ,Shell, Netto, School, Population, Water]])[0];
    Results['linearRegression'] = {};
    Results['linearRegression']['accuracy'] = lR['accuracy'];
    Results['linearRegression']['mse'] = lR['mse'];
    Results['linearRegression']['predict'] = lR['machine'].predict([[D, t, L, d, YS, UTS, Exp, B31G, MB31G, DNV, Battelle ,Shell, Netto, School, Population, Water]])[0];
    # Results['neuralNetwork'] = {};
    # Results['neuralNetwork']['accuracy'] = nN['accuracy'];
    # Results['neuralNetwork']['mse'] = nN['mse'];
    # Results['neuralNetwork']['predict'] = nN['machine'].predict([[D, t, L, d, YS, UTS, Exp, B31G, MB31G, DNV, Battelle ,Shell, Netto, School, Population, Water]])[0];
    Results['randomForest'] = {};
    Results['randomForest']['accuracy'] = rF['accuracy'];
    Results['randomForest']['mse'] = rF['mse'];
    Results['randomForest']['predict'] = rF['machine'].predict([[D, t, L, d, YS, UTS, Exp, B31G, MB31G, DNV, Battelle ,Shell, Netto, School, Population, Water]])[0];
    # Results['ridgeRegression'] = {};
    # Results['ridgeRegression']['accuracy'] = rR['accuracy'];
    # Results['ridgeRegression']['mse'] = rR['mse'];
    # Results['ridgeRegression']['predict'] = rR['machine'].predict([[D, t, L, d, YS, UTS, Exp, B31G, MB31G, DNV, Battelle ,Shell, Netto, School, Population, Water]])[0];
    Results['svm'] = {};
    Results['svm']['accuracy'] = sv['accuracy'];
    Results['svm']['mse'] = sv['mse'];
    Results['svm']['predict'] = sv['machine'].predict([[D, t, L, d, YS, UTS, Exp, B31G, MB31G, DNV, Battelle ,Shell, Netto, School, Population, Water]])[0];
    Results['Result'] = Result;

    Results['rangePercent'] = rangePercent;

    return json.dumps(Results, cls=NumpyArrayEncoder)

if __name__ == '__main__':
    app.run('0.0.0.0', port=5000, debug=True)