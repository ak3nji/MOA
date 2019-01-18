/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package fs.core.cost;

import fs.core.CostFunction;

import java.util.ArrayList;

/**
 *
 * @author liac01
 */
public class CostFunctionCorrelation implements CostFunction {

    private double[][] correlations;
    private int numAttributes;
    private int numClasses;
    private ArrayList<Double> corrClasses;

    public CostFunctionCorrelation(double[][] corr, int nAttributes, int nClasses)
    {
        this.correlations = corr;
        this.numAttributes = nAttributes;
        this.numClasses = nClasses;
        this.corrClasses = new ArrayList<>();
        this.mxCalculateCorrelationClasses();
    }

    private void mxCalculateCorrelationClasses()
    {
        double sumTmp;

        for (int i = 0; i < this.numAttributes; i++) {
            sumTmp = 0;
            for (int j = 0; j < this.numClasses; j++) {
                sumTmp += Math.abs(this.correlations[i][this.numAttributes + j]);
            }
            this.corrClasses.add(sumTmp / this.numClasses);
        }
        boolean llOk = true;
        for (int i = 1; i < this.numAttributes; i++) {
            if ((this.corrClasses.get(i - 1) - this.corrClasses.get(i)) != 0) {
                llOk = false;
                break;
            }
        }

        if (llOk) {
            for (int i = 0; i < this.numAttributes; i++) {
                this.corrClasses.set(i, 1.0);
            }
        }

    }

    @Override
    public double evaluateSubSet(ArrayList<Integer> subset)
    {
        double sumTmp, featClass = 0, featFeat = 0, merit;
        int numFeatures = subset.size();

        if (numFeatures == 0) {
            return 0;
        }

        //Feature-features correlation
        for (int i = 0; i < numFeatures; i++) {
            sumTmp = 0;
            for (int j = 0; j < numFeatures; j++) {
                if (i == j) {
                    continue;
                }
                sumTmp += Math.abs(this.correlations[subset.get(i)][subset.get(j)]);
            }
            //Average
            featFeat += ((sumTmp / (numFeatures - (numFeatures > 1 ? 1 : 0))) / numFeatures);
        }

        //Feature-class correlation
        for (int i = 0; i < numFeatures; i++) {
            featClass += this.corrClasses.get(subset.get(i)) / numFeatures;
        }

        //Heuristic "Merit" of a feature subset (Pearson's correlation)
        merit = (featClass * numFeatures)
            / (Math.sqrt(numFeatures + (numFeatures * (numFeatures - 1) * featFeat)));

        return merit;
    }

    @Override
    public double evaluateFeature(int feature)
    {
        double result, sumTmp = 0;

        //Feature correlation
        for (int j = 0; j < this.numAttributes; j++) {
            if (feature == j) {
                continue;
            }
            sumTmp += Math.abs(this.correlations[feature][j]);
        }
        sumTmp = sumTmp / (this.numAttributes - 1);

        //Class correlation / (feature correlation)
        result = this.corrClasses.get(feature) / sumTmp;

        return result;
    }
}
