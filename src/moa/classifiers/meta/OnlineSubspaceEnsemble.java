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
public class OnlineSubspaceEnsemble extends AbstractClassifier implements MultiClassClassifier {

    @Override
    public String getPurposeString() {
        return "Incremental on-line bagging of Oza and Russell.";
    }

    private static final long serialVersionUID = 1L;

    public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l', "Classifier to train.", Classifier.class,
            "trees.HoeffdingTree");

    public IntOption ensembleSizeOption = new IntOption("ensembleSize", 's', "The number of models in the bag.", 10, 1,
            Integer.MAX_VALUE);

    public IntOption chunkSizeOption = new IntOption("chunkSize", 'c',
            "The chunk size used for classifier creation and evaluation.", 10, 1, Integer.MAX_VALUE);

    public FloatOption subspaceSizeOption = new FloatOption("SubspaceSize", 'p',
            "Size of each subspace. Percentage of the number of attributes.", 0.5, 0.1, 1.0);

    public FloatOption lambdaOption = new FloatOption("lambda", 'a', "The lambda parameter for bagging.", 6.0, 1.0,
            Float.MAX_VALUE);
    
    public ClassOption driftDetectionMethodOption = new ClassOption("driftDetectionMethod", 'd',
            "Drift detection method to use.", ChangeDetector.class, "DDM");

    protected List<SubspaceLearner> ensemble;
    protected List<Double> weights;
    protected Instances buffer;
    protected int subspaceSize;
    protected ChangeDetector driftDetectionMethod;

    @Override
    public void resetLearningImpl() {
    	this.driftDetectionMethod = ((ChangeDetector) getPreparedClassOption(this.driftDetectionMethodOption)).copy();
        this.ensemble = null;
        this.buffer = null;
    }

    @Override
    public void trainOnInstanceImpl(Instance instance) {

        if (this.ensemble == null) {
            this.buildEnsemble(instance);
        }

        // Store instance in the buffer
        if (this.buffer == null) {
            this.buffer = new Instances(instance.dataset());
        }

        // Chunk is not full
        if (this.buffer.numInstances() != this.chunkSizeOption.getValue()) {
            this.buffer.add(instance);
        } else {
            // Feature Selection
            int[] newAttr = this.performFeatureSelection();
            Arrays.sort(newAttr);
            StringBuffer sb = new StringBuffer("");
            int n = 0;
            for (n = 0; n < newAttr.length - 2; n++) {
                sb.append((newAttr[n] + 1) + ",");
            }
            sb.append((newAttr[n] + 1));

            // Remove ensemble members
            int idx = 0, minCommom = this.ensemble.get(idx).commomAttributes(newAttr);
            for (int i = 0; i < this.ensemble.size(); i++) {
                int commomAttr = this.ensemble.get(i).commomAttributes(newAttr);
                if (commomAttr < minCommom) {
                    minCommom = commomAttr;
                    idx = i;
                }
            }
            this.removeEnsembleMemser(idx);
            // Add new expert
            Classifier baseLearner = (Classifier) getPreparedClassOption(this.baseLearnerOption);
            baseLearner.resetLearning();
            SubspaceLearner tmpClassifier = new SubspaceLearner(baseLearner.copy(), sb.toString(), new BasicClassificationPerformanceEvaluator());
            // Training classifier
            for (int i = 0; i < this.buffer.numInstances(); i++) {
                Instance trainInst = this.buffer.get(i);
                tmpClassifier.trainOnInstance(trainInst);
            }
            this.ensemble.add(tmpClassifier);
            this.weights.add(1.0 / (double) this.ensemble.size());
            this.buffer = new Instances(this.getModelContext());
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
    protected int testCounter = 0;
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
        /*testCounter++;
        double[] votes;
        if (testCounter % 1000 == 0){
        	System.out.println("Classe: " + instance.classValue());
        	votes = combinedVote.getArrayRef();
        	for (double votesClass: votes) {
        		System.out.println(votesClass);
        	}
        }*/
        return combinedVote.getArrayRef();
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
        return new Measurement[] { new Measurement("ensemble size", this.ensemble != null ? this.ensemble.size() : 0) };
    }

    private void buildEnsemble(Instance instance) {
        this.ensemble = new ArrayList<SubspaceLearner>();
        this.weights = new ArrayList<Double>();
        BasicClassificationPerformanceEvaluator classificationEvaluator = new BasicClassificationPerformanceEvaluator();
        Classifier baseLearner = (Classifier) getPreparedClassOption(this.baseLearnerOption);
        baseLearner.resetLearning();

        Integer[] indices = new Integer[instance.numAttributes() - 1];
        int classIndex = instance.classIndex();
        int offset = 0;
        for (int i = 0; i < indices.length + 1; i++) {
            if (i != classIndex) {
                indices[offset++] = i + 1;
            }
        }

        this.subspaceSize = this.numberOfAttributes(indices.length, this.subspaceSizeOption.getValue());
        for (int i = 0; i < this.ensembleSizeOption.getValue(); i++) {
            SubspaceLearner tmpClassifier = new SubspaceLearner(baseLearner.copy(),
                    this.randomSubSpace(indices, this.subspaceSize, classIndex + 1),
                    (BasicClassificationPerformanceEvaluator) classificationEvaluator.copy());
            this.ensemble.add(tmpClassifier);
            this.weights.add(1.0 / (double) this.ensemble.size());
        }
    }

    protected int numberOfAttributes(int total, double fraction) {
        int k = (int) Math.round((fraction < 1.0) ? total * fraction : fraction);

        if (k > total)
            k = total;
        if (k < 1)
            k = 1;

        return k;
    }

    protected String randomSubSpace(Integer[] indices, int subSpaceSize, int classIndex) {
        Collections.shuffle(Arrays.asList(indices), this.classifierRandom);
        StringBuffer sb = new StringBuffer("");
        int i = 0;
        for (i = 0; i < subSpaceSize - 1; i++) {
            sb.append(indices[i] + ",");
        }
        sb.append(indices[i]);

        return sb.toString();
    }

    private int[] performFeatureSelection() {
        SamoaToWekaInstanceConverter convToWeka = new SamoaToWekaInstanceConverter();
        weka.core.Instances wbuffer = convToWeka.wekaInstances(this.buffer);
        AttributeSelection attsel = new AttributeSelection();
        SymmetricalUncertAttributeSetEval evaluator = new SymmetricalUncertAttributeSetEval();
        int[] newAttr = null;

        attsel.setEvaluator(evaluator);
        FCBFSearch search = new FCBFSearch();
        search.setThreshold(0);
        attsel.setSearch(search);

        try {
            attsel.SelectAttributes(wbuffer);
            newAttr = attsel.selectedAttributes();
        } catch (Exception ex) {
        }

        return newAttr;
    }

    protected void removeEnsembleMemser(int i) {
        this.ensemble.remove(i);
        this.weights.remove(i);
    }

    protected final class SubspaceLearner extends AbstractMOAObject {

        private static final long serialVersionUID = 1L;

        protected InstancesHeader streamHeader;
        protected Classifier classifier;
        protected Selection inputsSelected;
        protected List<Integer> filteredAttributes;
        protected String inputString;
        public BasicClassificationPerformanceEvaluator evaluator;

        public SubspaceLearner(Classifier classifier, String inputString,
                BasicClassificationPerformanceEvaluator evaluatorInstantiated) {
            this.classifier = classifier;
            this.inputString = inputString;
            this.evaluator = evaluatorInstantiated;
        }

        public void reset() {
            this.classifier.resetLearning();
            this.streamHeader = null;
        }

        public void trainOnInstance(Instance instance) {
            if (this.streamHeader == null) {
                this.streamHeader = this.constructHeader(instance);
            }

            this.classifier.trainOnInstance(this.filterInstance(instance));
        }

        public double[] getVotesForInstance(Instance instance) {
            if (this.streamHeader == null) {
                this.streamHeader = this.constructHeader(instance);
            }
            return this.classifier.getVotesForInstance(this.filterInstance(instance));
        }

        private Instance filterInstance(Instance instance) {
            double[] attValues = new double[this.streamHeader.numAttributes()];
            Instance newInstance = new InstanceImpl(instance.weight(), attValues);

            int count = 0;
            for (int i = 0; i < inputsSelected.numEntries(); i++) {
                int start = inputsSelected.getStart(i) - 1;
                int end = inputsSelected.getEnd(i) - 1;
                for (int j = start; j <= end; j++) {
                    newInstance.setValue(count, instance.value(j));
                    count++;
                }
            }

            newInstance.setValue(count, instance.classValue());
            newInstance.setDataset(this.streamHeader);

            return newInstance;
        }

        private InstancesHeader constructHeader(Instance instance) {
            inputsSelected = getSelection(inputString);

            int totAttributes = inputsSelected.numValues() + 1;
            Instances ds = new Instances();
            List<Attribute> v = new ArrayList<Attribute>(totAttributes);
            List<Integer> indexValues = new ArrayList<Integer>(totAttributes);
            int ct = 0;

            // Selected input values
            for (int i = 0; i < inputsSelected.numEntries(); i++) {
                for (int j = inputsSelected.getStart(i); j <= inputsSelected.getEnd(i); j++) {
                    v.add(instance.attribute(j - 1));
                    indexValues.add(ct);
                    ct++;
                }
            }
            // Filtered Attributes
            this.filteredAttributes = new ArrayList<Integer>(indexValues);
            Collections.sort(filteredAttributes);

            // Class value
            v.add(instance.attribute(instance.classIndex()));
            indexValues.add(ct);

            ds.setAttributes(v, indexValues);
            Range r = new Range("-" + 1);
            r.setUpper(totAttributes);
            ds.setRangeOutputIndices(r);

            return (new InstancesHeader(ds));
        }

        private Selection getSelection(String text) {
            Selection s = new Selection();
            String[] parts = text.trim().split(",");
            for (String p : parts) {
                int index = p.indexOf('-');
                if (index == -1) {// is a single entry
                    s.add(Integer.parseInt(p));
                } else {
                    String[] vals = p.split("-");
                    s.add(Integer.parseInt(vals[0]), Integer.parseInt(vals[1]));
                }
            }
            return s;
        }

        public int commomAttributes(int[] attributes) {
            List<Integer> listAttributes = new ArrayList<Integer>();
            for (int i : attributes) {
                listAttributes.add(i);
            }
            listAttributes.retainAll(this.filteredAttributes);

            return listAttributes.size();
        }

        @Override
        public void getDescription(StringBuilder arg0, int arg1) {

        }
    }
}