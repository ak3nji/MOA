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

import moa.AbstractMOAObject;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.MultiClassClassifier;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.core.MiscUtils;
import moa.options.ClassOption;
import moa.streams.filters.Selection;

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
 * [OR] N. Oza and S. Russell. Online bagging and boosting. In Artiﬁcial
 * Intelligence and Statistics 2001, pages 105–112. Morgan Kaufmann, 2001.
 * </p>
 *
 * <p>
 * Parameters:
 * </p>
 * <ul>
 * <li>-l : Classiﬁer to train</li>
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

    public FloatOption subspaceSizeOption = new FloatOption("SubspaceSize", 'p',
            "Size of each subspace. Percentage of the number of attributes.", 0.7, 0.1, 1.0);

    protected SubspaceLearner[] ensemble;

    @Override
    public void resetLearningImpl() {
        this.ensemble = null;
    }

    @Override
    public void trainOnInstanceImpl(Instance instance) {

        if (this.ensemble == null) {
            this.buildEnsemble(instance);
        }

        for (int i = 0; i < this.ensemble.length; i++) {
            int k = MiscUtils.poisson(1.0, this.classifierRandom);
            if (k > 0) {
                Instance weightedInst = (Instance) instance.copy();
                weightedInst.setWeight(instance.weight() * k);
                this.ensemble[i].trainOnInstance(weightedInst);
            }
        }
    }

    @Override
    public double[] getVotesForInstance(Instance instance) {

        if (this.ensemble == null) {
            this.buildEnsemble(instance);
        }

        DoubleVector combinedVote = new DoubleVector();
        for (int i = 0; i < this.ensemble.length; i++) {
            DoubleVector vote = new DoubleVector(this.ensemble[i].getVotesForInstance(instance));
            if (vote.sumOfValues() > 0.0) {
                vote.normalize();
                combinedVote.addValues(vote);
            }
        }
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
        return new Measurement[] { new Measurement("ensemble size", this.ensemble != null ? this.ensemble.length : 0) };
    }

    private void buildEnsemble(Instance instance) {
        this.ensemble = new SubspaceLearner[this.ensembleSizeOption.getValue()];
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

        int subSpaceSize = this.numberOfAttributes(indices.length, this.subspaceSizeOption.getValue());
        for (int i = 0; i < this.ensemble.length; i++) {
            SubspaceLearner tmpClassifier = new SubspaceLearner(baseLearner.copy(),
                    this.randomSubSpace(indices, subSpaceSize, classIndex + 1));
            this.ensemble[i] = tmpClassifier;
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

    protected final class SubspaceLearner extends AbstractMOAObject {

        private static final long serialVersionUID = 1L;

        protected InstancesHeader streamHeader;
        protected Classifier classifier;
        protected List<Integer> filteredAttributes;
        protected Selection inputsSelected;
        protected String inputString;

        public SubspaceLearner(Classifier classifier, String inputString) {
            this.classifier = classifier;
            this.inputString = inputString;
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

        @Override
        public void getDescription(StringBuilder arg0, int arg1) {

        }
    }
}