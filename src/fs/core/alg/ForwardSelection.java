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
public class ForwardSelection extends AlgorithmSelection {

    /**
     * Construct for forward selection algorithm
     *
     * @param numFeatures
     * @param costFunction
     */
    public ForwardSelection(int numFeatures, CostFunction costFunction)
    {
        this.numFeatures = numFeatures;
        this.selectFeatures = new boolean[this.numFeatures];
        this.subSetFeatures = new ArrayList<>();
        this.costFunction = costFunction;
    }

    @Override
    public ArrayList<Integer> algorithm(int numBestFeatures)
    {
        double maxValue, value;
        int indexFeature = 0;

        while (this.subSetFeatures.size() < numBestFeatures) {
            maxValue = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < this.numFeatures; i++) {
                if (!this.selectFeatures[i]) {
                    this.subSetFeatures.add(i);
                    value = this.costFunction.evaluateSubSet(this.subSetFeatures); //more
                    if (value > maxValue) {
                        maxValue = value;
                        indexFeature = i;
                    }
                    this.subSetFeatures.remove(this.subSetFeatures.size() - 1);
                }
            }
            this.addFeatureSubSet(indexFeature);
        }

        return this.subSetFeatures;
    }

    @Override
    public ArrayList<Integer> algorithm()
    {
        double maxValue = Double.NEGATIVE_INFINITY, value, bestValue;
        int indexFeature = 0;

        do {
            bestValue = maxValue;
            maxValue = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < this.numFeatures; i++) {
                if (!this.selectFeatures[i]) {
                    this.subSetFeatures.add(i);
                    value = this.costFunction.evaluateSubSet(this.subSetFeatures); //more
                    if (value > maxValue) {
                        maxValue = value;
                        indexFeature = i;
                    }
                    this.subSetFeatures.remove(this.subSetFeatures.size() - 1);
                }
            }
            if (maxValue >= bestValue) {
                this.addFeatureSubSet(indexFeature);
            }
        } while (maxValue >= bestValue && this.subSetFeatures.size() <= this.numFeatures);

        return this.subSetFeatures;
    }

    @Override
    public ArrayList<Integer> algorithm(double threshold)
    {
        double maxValue = Double.NEGATIVE_INFINITY, value, bestValue = Double.NEGATIVE_INFINITY;
        int indexFeature = 0;

        do {
            if (maxValue > bestValue) {
                bestValue = maxValue;
            }
            maxValue = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < this.numFeatures; i++) {
                if (!this.selectFeatures[i]) {
                    this.subSetFeatures.add(i);
                    value = this.costFunction.evaluateSubSet(this.subSetFeatures); //more
                    if (value > maxValue) {
                        maxValue = value;
                        indexFeature = i;
                    }
                    this.subSetFeatures.remove(this.subSetFeatures.size() - 1);
                }
            }
            if (maxValue >= (bestValue * threshold)) {
                this.addFeatureSubSet(indexFeature);
            }
        } while (maxValue >= (bestValue * threshold) && this.subSetFeatures.size() <= this.numFeatures);

        return this.subSetFeatures;
    }
}
