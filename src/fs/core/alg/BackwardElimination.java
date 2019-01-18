/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package fs.core.alg;

import java.util.ArrayList;

import fs.core.AlgorithmSelection;
import fs.core.CostFunction;

/**
 *
 * @author liac01
 */
public class BackwardElimination extends AlgorithmSelection {

    /**
     * Construct for backward elimination algorithm
     *
     * @param numFeatures
     * @param costFunction
     */
    public BackwardElimination(int numFeatures, CostFunction costFunction)
    {
        this.numFeatures = numFeatures;
        this.selectFeatures = new boolean[this.numFeatures];
        this.subSetFeatures = new ArrayList<>();
        this.costFunction = costFunction;

        for (int i = 0; i < this.numFeatures; i++) {
            this.selectFeatures[i] = true;
            this.subSetFeatures.add(i);
        }
    }

    @Override
    public ArrayList<Integer> algorithm(int numBestFeatures)
    {
        double maxValue, value;
        int indexFeature = 0;

        while (this.subSetFeatures.size() > numBestFeatures) {
            maxValue = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < this.numFeatures; i++) {
                if (this.selectFeatures[i]) {
                    this.subSetFeatures.remove(new Integer(i));
                    value = this.costFunction.evaluateSubSet(this.subSetFeatures); //more
                    if (value > maxValue) {
                        maxValue = value;
                        indexFeature = i;
                    }
                    this.subSetFeatures.add(i);
                }
            }
            this.removeFeatureSubSet(indexFeature);
        }

        return this.subSetFeatures;
    }

    @Override
    public ArrayList<Integer> algorithm()
    {
        double maxValue = Double.NEGATIVE_INFINITY, value, bestValue;
        int indexFeature = 0;

        maxValue = this.costFunction.evaluateSubSet(this.subSetFeatures);
        do {
            bestValue = maxValue;
            maxValue = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < this.numFeatures; i++) {
                if (this.selectFeatures[i]) {
                    this.subSetFeatures.remove(new Integer(i));
                    value = this.costFunction.evaluateSubSet(this.subSetFeatures);
                    if (value > maxValue) {
                        maxValue = value;
                        indexFeature = i;
                    }
                    this.subSetFeatures.add(i);
                }
            }
            if (maxValue > bestValue && this.subSetFeatures.size() > 1) {
                this.removeFeatureSubSet(indexFeature);
            }
        } while (maxValue > bestValue && this.subSetFeatures.size() > 1);

        return this.subSetFeatures;
    }

    @Override
    public ArrayList<Integer> algorithm(double threshold)
    {
        double maxValue = Double.NEGATIVE_INFINITY, value, bestValue;
        int indexFeature = 0;

        maxValue = this.costFunction.evaluateSubSet(this.subSetFeatures);
        do {
            bestValue = maxValue;
            maxValue = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < this.numFeatures; i++) {
                if (this.selectFeatures[i]) {
                    this.subSetFeatures.remove(new Integer(i));
                    value = this.costFunction.evaluateSubSet(this.subSetFeatures);
                    if (value > maxValue) {
                        maxValue = value;
                        indexFeature = i;
                    }
                    this.subSetFeatures.add(i);
                }
            }
            if ((maxValue * threshold) > bestValue && this.subSetFeatures.size() > 1) {
                this.removeFeatureSubSet(indexFeature);
            }
        } while ((maxValue * threshold) > bestValue && this.subSetFeatures.size() > 1);

        return this.subSetFeatures;
    }
}