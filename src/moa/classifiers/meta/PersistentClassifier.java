package moa.classifiers.meta;

import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.MultiClassClassifier;
import moa.core.DoubleVector;
import moa.core.Measurement;


public class PersistentClassifier extends AbstractClassifier implements MultiClassClassifier {
	
	

	private static final long serialVersionUID = 1L;
	
	protected int previousClass;
	
	protected double[] makeVotesArray() {
		DoubleVector votesArray = new DoubleVector();

		votesArray.addToValue(this.previousClass,1);

		return votesArray.getArrayCopy();
	}
	
	public boolean isRandomizable() {
		return false;
	}
	
	@Override
    public void getModelDescription(StringBuilder out, int indent) {
        out.append("Predicts the new instance's class as the last seen class.");
    }
	
	@Override
    protected Measurement[] getModelMeasurementsImpl() {
        Measurement[] measurements = null;
        return measurements;
    }
	
	@Override
	public void resetLearningImpl() {
	    this.previousClass = 0;
	}
	
	@Override
	public void trainOnInstanceImpl(Instance inst) {
	    this.previousClass = (int)inst.classValue();
	}
	
	protected int testCounter = 0;
	
	public double[] getVotesForInstance(Instance inst) {
		testCounter++;
//        double[] votes;
//        if (testCounter % 1000 == 0){
//        	System.out.println("Class: " + inst.classValue());
//        	votes = makeVotesArray();
//        	System.out.println("Total classes = "+votes.length);
//        	for (double votesClass: votes) {
//        		System.out.println(votesClass);
//        	}
//        }
		return makeVotesArray();

	}

}
