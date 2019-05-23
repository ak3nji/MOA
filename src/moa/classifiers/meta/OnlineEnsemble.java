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
import moa.classifiers.meta.OnlineSubspaceEnsemble.SubspaceLearner;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.core.MiscUtils;
import moa.core.Utils;
import moa.evaluation.BasicClassificationPerformanceEvaluator;
import moa.options.ClassOption;
import moa.streams.filters.Selection;

public class OnlineEnsemble extends AbstractClassifier implements MultiClassClassifier {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
    public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l', "Classifier to train.", Classifier.class,
            "trees.HoeffdingTree");
    
    public IntOption ensembleSizeOption = new IntOption("ensembleSize", 's', "The number of models in the bag.", 10, 1,
            Integer.MAX_VALUE);
    
    public FloatOption lambdaOption = new FloatOption("lambda", 'a', "The lambda parameter for bagging.", 6.0, 1.0,
            Float.MAX_VALUE);
    
    protected List<Learner> ensemble;
    protected List<Double> weights;
    
    @Override
    public String getPurposeString() {
        return "Incremental on-line bagging of Oza and Russell.";
    }

	@Override
	public boolean isRandomizable() {
		return true;
	}

	@Override
	public void resetLearningImpl() {
		this.ensemble = null;
		
	}

	@Override
	public void trainOnInstanceImpl(Instance instance) {
		if (this.ensemble == null) {
            this.buildEnsemble(instance);
        }
		
		for (int i = 0; i < this.ensemble.size(); i++) {
            int k = MiscUtils.poisson(this.lambdaOption.getValue(), this.classifierRandom);
            if (k > 0) {
                Instance weightedInst = (Instance) instance.copy();
                weightedInst.setWeight(instance.weight() * k);
                this.ensemble.get(i).trainOnInstance(weightedInst);
            }
        }
		
	}

	@Override
	public double[] getVotesForInstance(Instance instance) {

        if (this.ensemble == null) {
            this.buildEnsemble(instance);
        }

        DoubleVector combinedVote = new DoubleVector();
        for (int i = 0; i < this.ensemble.size(); i++) {
            DoubleVector vote = new DoubleVector(this.ensemble.get(i).getVotesForInstance(instance));
            if (vote.sumOfValues() > 0.0) {
                vote.normalize();
                vote.scaleValues(this.weights.get(i));
                combinedVote.addValues(vote);
            }
        }

        return combinedVote.getArrayRef();
    }
	
	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		return null;
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {
		
	}
	
	private void buildEnsemble(Instance instance) {
        this.ensemble = new ArrayList<Learner>();
        this.weights = new ArrayList<Double>();
        BasicClassificationPerformanceEvaluator classificationEvaluator = new BasicClassificationPerformanceEvaluator();
        Classifier baseLearner = (Classifier) getPreparedClassOption(this.baseLearnerOption);
        baseLearner.resetLearning();

        for (int i = 0; i < this.ensembleSizeOption.getValue(); i++) {
            Learner tmpClassifier = new Learner(baseLearner.copy(),
                    (BasicClassificationPerformanceEvaluator) classificationEvaluator.copy());
            this.ensemble.add(tmpClassifier);
            this.weights.add(1.0 / (double) this.ensemble.size());
        }
    }
	
	protected final class Learner extends AbstractMOAObject {

        private static final long serialVersionUID = 1L;

        protected Classifier classifier;
        public BasicClassificationPerformanceEvaluator evaluator;

        public Learner(Classifier classifier,
                BasicClassificationPerformanceEvaluator evaluatorInstantiated) {
            this.classifier = classifier;
            this.evaluator = evaluatorInstantiated;
        }

        public void reset() {
            this.classifier.resetLearning();
        }

        public void trainOnInstance(Instance instance) {

            this.classifier.trainOnInstance(instance);
        }

        public double[] getVotesForInstance(Instance instance) {
        	
            return this.classifier.getVotesForInstance(instance);
        }




        @Override
        public void getDescription(StringBuilder arg0, int arg1) {

        }
    }

}
