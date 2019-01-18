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
public class FloatingForwardSelection extends AlgorithmSelection {

    private int currentIndex;

    /**
     * Construct for floating forward selection algorithm
     *
     * @param numFeatures
     * @param costFunction
     */
    public FloatingForwardSelection(int numFeatures, CostFunction costFunction)
    {
        this.numFeatures = numFeatures;
        this.selectFeatures = new boolean[this.numFeatures];
        this.subSetFeatures = new ArrayList<>();
        this.costFunction = costFunction;
    }

    @Override
    public ArrayList<Integer> algorithm(int numBestFeatures)
    {
        double maxValue, currentValue;
        int indexFeature;
        int k = 0;
        ArrayList<Double> values = new ArrayList<>();

        while (k < numBestFeatures) {
            currentValue = this.mostSignificantFeature();
            indexFeature = this.currentIndex;

            if (k < 2) {
                this.addFeatureSubSet(indexFeature);
                //Save J(X)
                values.add(currentValue);
                k = k + 1;
            } else {
                // Step 1: Inclusion
                // Temporally add the most significant feature x_k+1 to X_k, to format X_k+1
                this.addFeatureSubSet(indexFeature);

                // Step 2: Conditional Exclusion
                // Find the least significant feature in the set X_k+1
                maxValue = this.leastSignificantFeature();
                //If the least significant feature in the set X is x_k+1
                if (maxValue <= values.get(k - 1)) {
                    values.add(currentValue);
                    k = k + 1;
                } else {
                    currentValue = maxValue;
                    // Exclude "currentIndex" (x_r) from X_k+1, to format X'_k
                    this.removeFeatureSubSet(this.currentIndex);
                    if (k == 2) {
                        // Update the value for the set with k features
                        values.set(k - 1, currentValue);
                    } // Step 3: Continuation of conditional Exclusion
                    else {
                        while (k > 2) {
                            // Find the least significant feature (x_s) in the set X'_k
                            maxValue = this.leastSignificantFeature();
                            if (maxValue <= values.get(k - 2)) {
                                // Update the value for the set with k features						
                                values.set(k - 1, currentValue);
                                break;
                            } else {
                                currentValue = maxValue;
                                // Exclude "currentIndex" (x_s) from X'_k, to format X'_k-1
                                this.removeFeatureSubSet(this.currentIndex);
                                k = k - 1;
                                if (k == 2) {
                                    // Update the value for the set with k features
                                    values.set(k - 1, currentValue);
                                }
                            }
                        }
                    }
                }
            }
        }

        return this.subSetFeatures;
    }

    @Override
    public ArrayList<Integer> algorithm()
    {
        return null;
    }

    @Override
    public ArrayList<Integer> algorithm(double threshold)
    {
        return null;
    }

    private double mostSignificantFeature()
    {
        double value, maxValue = Double.NEGATIVE_INFINITY;
        int indexFeature = -1;

        for (int i = 0; i < this.numFeatures; i++) {
            if (!this.selectFeatures[i]) {
                this.subSetFeatures.add(i);
                value = this.costFunction.evaluateSubSet(this.subSetFeatures); // less
                if (value > maxValue) {
                    maxValue = value;
                    indexFeature = i;
                }
                this.subSetFeatures.remove(this.subSetFeatures.size() - 1);
            }
        }
        this.currentIndex = indexFeature;

        return maxValue;
    }

    private double leastSignificantFeature()
    {
        double value, maxValue = Double.NEGATIVE_INFINITY;
        int indexFeature = -1;

        for (int i = 0; i < this.numFeatures; i++) {
            if (this.selectFeatures[i]) {
                this.subSetFeatures.remove((Integer) i);
                value = this.costFunction.evaluateSubSet(this.subSetFeatures); // less
                if (value > maxValue) {
                    maxValue = value;
                    indexFeature = i;
                }
                this.subSetFeatures.add(i);
            }
        }
        this.currentIndex = indexFeature;

        return maxValue;
    }
}
