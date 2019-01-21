package moa.classifiers.meta;

import java.util.ArrayList;
import java.util.List;

import com.github.javacliparser.StringOption;
import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstanceImpl;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import com.yahoo.labs.samoa.instances.Range;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.MultiClassClassifier;
import moa.core.Measurement;
import moa.options.ClassOption;
import moa.streams.filters.Selection;

/**
 * Class for running an arbitrary classifier on data that has been passed
 * through an arbitrary filter.
 * <p>
 *
 * Valid options are:
 * <p>
 *
 * -l classname <br>
 * Specify the full class name of a classifier as the basis for the concept
 * drift classifier.
 *
 * @author Jorge Chamby-Diaz (jchambyd at gmail dot com)
 * @version $Revision: 1 $
 */
public class FilteredClassifier extends AbstractClassifier implements MultiClassClassifier {

    private static final long serialVersionUID = 1L;

    @Override
    public String getPurposeString() {
        return "Class for running an arbitrary classifier on data that has been passed through an arbitrary filter.";
    }

    public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l', "Classifier to train.", Classifier.class,
            "bayes.NaiveBayes");

    public StringOption inputStringOption = new StringOption("inputStringOption", 'i',
            "Selection of attributes to be used as input.", "1");

    protected InstancesHeader streamHeader;
    protected Classifier classifier;
    protected List<Integer> filteredAttributes;
    protected Selection inputsSelected;

    @Override
    public void resetLearningImpl() {
        this.classifier = ((Classifier) getPreparedClassOption(this.baseLearnerOption));
        this.classifier.resetLearning();
        this.streamHeader = null;
    }

    @Override
    public void trainOnInstanceImpl(Instance instance) {
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

    @Override
    public boolean isRandomizable() {
        return true;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
        ((AbstractClassifier) this.classifier).getModelDescription(out, indent);
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        Measurement[] measurements = null;
        return measurements;
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
        inputsSelected = getSelection(inputStringOption.getValue());

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
}