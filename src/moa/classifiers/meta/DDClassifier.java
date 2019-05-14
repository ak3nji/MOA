package moa.classifiers.meta;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstanceImpl;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import com.yahoo.labs.samoa.instances.Range;
import com.yahoo.labs.samoa.instances.SamoaToWekaInstanceConverter;

import moa.AbstractMOAObject;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.core.driftdetection.ChangeDetector;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.core.MiscUtils;
import moa.core.Utils;
import moa.evaluation.BasicClassificationPerformanceEvaluator;
import moa.options.ClassOption;
import moa.streams.filters.Selection;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.FCBFSearch;
import weka.attributeSelection.SymmetricalUncertAttributeSetEval;

/**
 * Incremental on-line bagging of Oza and Russell.
 *
 * <p>
 * Oza and Russell developed online versions of bagging and boosting for Data
 * Streams. They show how the process of sampling bootstrap replicates from
 * training data can be simulated in a data stream context. They observe that
 * the probability that any individual example will be chosen for a replicate
 * tends to a Poisson(1) distribution.
 * </p>
 *
 * <p>
 * [OR] N. Oza and S. Russell. Online bagging and boosting. 
 * Intelligence and Statistics 2001, pages 105–112. Morgan Kaufmann, 2001.
 * </p>
 *
 * <p>
 * Parameters:
 * </p>
 * <ul>
 * <li>-l :  to train</li>
 * <li>-s : The number of models in the bag</li>
 * </ul>
 *
 * @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 * @version $Revision: 7 $
 */
public class DDClassifier extends AbstractClassifier implements MultiClassClassifier {

    @Override
    public String getPurposeString() {
        return "Incremental on-line bagging of Oza and Russell.";
    }

    private static final long serialVersionUID = 1L;

    public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l', "Classifier to train.", Classifier.class,
            "trees.RandomHoeffdingTree");
    
    public ClassOption driftDetectionMethodOption = new ClassOption("driftDetectionMethod", 'd',
            "Drift detection method to use.", ChangeDetector.class, "DDM");

    protected ChangeDetector driftDetectionMethod;
    protected Classifier classifier;
    protected int ddmLevel;
    protected boolean isDriftDetected;
    public boolean isWarningDetected() {
        return (this.ddmLevel == DDM_WARNING_LEVEL);
    }

    public boolean isChangeDetected() {
        return (this.ddmLevel == DDM_OUTCONTROL_LEVEL);
    }

    public static final int DDM_INCONTROL_LEVEL = 0;

    public static final int DDM_WARNING_LEVEL = 1;

    public static final int DDM_OUTCONTROL_LEVEL = 2;

    @Override
    public void resetLearningImpl() {
    	this.driftDetectionMethod = ((ChangeDetector) getPreparedClassOption(this.driftDetectionMethodOption)).copy();
    	this.classifier = ((Classifier) getPreparedClassOption(this.baseLearnerOption)).copy();
        this.classifier.resetLearning();
        this.isDriftDetected = false;
    }
    
    protected int changesDetected = 0;
    protected int ensembleUpdates = 0;

    
    @Override
    public void trainOnInstanceImpl(Instance instance) {
    	
    	
    	int trueClass = (int)instance.classValue();
    	
    	boolean prediction;
        if (Utils.maxIndex(getVotesForInstance(instance)) == trueClass) {
            prediction = true;
        } else {
            prediction = false;
        }
        
        this.driftDetectionMethod.input(prediction ? 0.0 : 1.0);
        this.ddmLevel = DDM_INCONTROL_LEVEL;
        if (this.driftDetectionMethod.getChange()) {
         this.ddmLevel =  DDM_OUTCONTROL_LEVEL;
        }
        if (this.driftDetectionMethod.getWarningZone()) {
           this.ddmLevel =  DDM_WARNING_LEVEL;
        }
        
        
        // Add instance to buffer or discard it.
    	switch(this.ddmLevel) {
    		case DDM_WARNING_LEVEL:

    			break;
    		case DDM_OUTCONTROL_LEVEL:
    			this.classifier.resetLearning();
    			break;
    		case DDM_INCONTROL_LEVEL:
    			break;
    	}
        

        this.classifier.trainOnInstance(instance);
    }
    protected int testCounter = 0;
    @Override
    public double[] getVotesForInstance(Instance instance) {

    	return this.classifier.getVotesForInstance(instance);
    }

    @Override
    public boolean isRandomizable() {
        return true;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return null;
    }

}