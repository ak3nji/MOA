/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package fs.core;

import java.util.ArrayList;

/**
 *
 * @author liac01
 */
public abstract class AlgorithmSelection {

    protected ArrayList<Integer> subSetFeatures;
    protected boolean[] selectFeatures;
    protected int numFeatures;
    protected CostFunction costFunction;

    public abstract ArrayList<Integer> algorithm(int numBestFeatures);

    public abstract ArrayList<Integer> algorithm();

    public abstract ArrayList<Integer> algorithm(double threshold);

    protected void addFeatureSubSet(int feature) {
        this.selectFeatures[feature] = true;
        this.subSetFeatures.add(feature);
    }

    protected void removeFeatureSubSet(int feature) {
        this.selectFeatures[feature] = false;
        this.subSetFeatures.remove(new Integer(feature));
    }

    /**
     * @return the subSetFeatures
     */
    protected ArrayList<Integer> getSubSetFeatures() {
        return subSetFeatures;
    }

    /**
     * @param subSetFeatures the subSetFeatures to set
     */
    protected void setSubSetFeatures(ArrayList<Integer> subSetFeatures) {
        this.subSetFeatures = subSetFeatures;
    }

    /**
     * @return the selectFeatures
     */
    protected boolean[] getSelectFeatures() {
        return selectFeatures;
    }

    /**
     * @param selectFeatures the selectFeatures to set
     */
    protected void setSelectFeatures(boolean[] selectFeatures) {
        this.selectFeatures = selectFeatures;
    }

    /**
     * @return the numFeatures
     */
    protected int getNumFeatures() {
        return numFeatures;
    }

    /**
     * @param numFeatures the numFeatures to set
     */
    protected void setNumFeatures(int numFeatures) {
        this.numFeatures = numFeatures;
    }

    /**
     * @return the costFunction
     */
    protected CostFunction getCostFunction() {
        return costFunction;
    }

    /**
     * @param costFunction the costFunction to set
     */
    protected void setCostFunction(CostFunction costFunction) {
        this.costFunction = costFunction;
    }
}