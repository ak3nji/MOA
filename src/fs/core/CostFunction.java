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
public interface CostFunction {

    public double evaluateSubSet(ArrayList<Integer> subset);

    public double evaluateFeature(int feature);
}
