package moa.classifiers.meta;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.SamoaToWekaInstanceConverter;
import com.yahoo.labs.samoa.instances.WekaToSamoaInstanceConverter;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.LFDD;
import moa.classifiers.MultiClassClassifier;
import moa.core.DoubleVector;
import moa.core.InstanceExample;
import moa.core.Measurement;
import moa.core.MiscUtils;
import moa.evaluation.BasicClassificationPerformanceEvaluator;
import moa.options.ClassOption;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.FCBFSearch;
import weka.attributeSelection.SymmetricalUncertAttributeSetEval;

/**
 *
 * @author Jorge C. Chamby-Diaz
 *
 *
 */
public class HEFT extends AbstractClassifier implements MultiClassClassifier {

    private static final long serialVersionUID = 1L;

    public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l', "Classifier to train.", Classifier.class,
            "bayes.NaiveBayes");

    public IntOption chunkSizeOption = new IntOption("chunkSize", 'c',
            "The chunk size used for classifier creation and evaluation.", 1000, 1, Integer.MAX_VALUE);

    public IntOption ensembleSizeOption = new IntOption("ensembleSize", 'e', "Ensemble size.", 10, 1,
            Integer.MAX_VALUE);

    public FloatOption lambdaOption = new FloatOption("lambda", 'a', "The lambda parameter for bagging.", 6.0, 1.0,
            Float.MAX_VALUE);

    protected List<Classifier> ensemble;
    protected List<AttributeSelection> attSelectors;
    protected List<BasicClassificationPerformanceEvaluator> evaluators;
    protected Instances buffer;
    protected weka.core.Instances wbuffer;
    protected int ensembleSize;
    protected int[] currentAttr;
    protected SamoaToWekaInstanceConverter convToWeka;
    protected WekaToSamoaInstanceConverter convToMoa;

    @Override
    public void resetLearningImpl() {
        this.ensemble = new ArrayList<>();
        this.evaluators = new ArrayList<>();
        this.attSelectors = new ArrayList<>();
        this.buffer = null;
        this.ensembleSize = this.ensembleSizeOption.getValue();
        this.convToWeka = new SamoaToWekaInstanceConverter();
        this.currentAttr = new int[0];
        this.ensemble.add(((Classifier) getPreparedClassOption(this.baseLearnerOption)));
        this.ensemble.get(0).resetLearning();
        this.evaluators.add(new BasicClassificationPerformanceEvaluator());
        this.attSelectors.add(null);
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        // Store instance in the buffer
        if (this.buffer == null) {
            this.buffer = new Instances(inst.dataset());
        }

        Instance trainInst;

        try {

            // Chunk is not full
            if (this.buffer.numInstances() != this.chunkSizeOption.getValue()) {
                this.buffer.add(inst);
            } else {
                // Feature Selection performed over data chunck
                AttributeSelection attrSel = this.performFeatureSelection();
                int[] newAttr = attrSel.selectedAttributes();
                Arrays.sort(newAttr);

                if (!Arrays.equals(this.currentAttr, newAttr)) {
                    // Building a new classifier associated with the new feature subset
                    Classifier classifier = ((Classifier) getPreparedClassOption(this.baseLearnerOption));
                    classifier.resetLearning();

                    this.convToMoa = new WekaToSamoaInstanceConverter();
                    for (weka.core.Instance winst : this.wbuffer) {
                        trainInst = this.convToMoa.samoaInstance(attrSel.reduceDimensionality(winst));
                        classifier.trainOnInstance(trainInst);
                    }

                    // Pruning classifier
                    if (this.ensemble.size() >= this.ensembleSize) {

                        int worstClassfier = 0;
                        double worstAcc = this.evaluators.get(0).getPerformanceMeasurements()[1].getValue();
                        for (int i = 1; i < this.ensemble.size(); i++) {
                            double tmpAcc = this.evaluators.get(i).getPerformanceMeasurements()[1].getValue();
                            if (tmpAcc < worstAcc) {
                                worstAcc = tmpAcc;
                                worstClassfier = i;
                            }
                        }
                        this.ensemble.remove(worstClassfier);
                        this.attSelectors.remove(worstClassfier);
                        this.evaluators.remove(worstClassfier);
                    }

                    // Add the new created classifier
                    this.ensemble.add(classifier);
                    this.attSelectors.add(attrSel);
                    this.evaluators.add(new BasicClassificationPerformanceEvaluator());
                    this.currentAttr = newAttr;
                }

                // Update m times each classifier with each instance x in chunk, according to
                // Poisson
                for (int i = 0; i < this.ensemble.size(); i++) {
                    this.evaluators.get(i).reset();
                    for (weka.core.Instance winst : this.wbuffer) {
                        int k = MiscUtils.poisson(this.lambdaOption.getValue(), this.classifierRandom);
                        this.convToMoa = new WekaToSamoaInstanceConverter();
                        // All attributes?
                        if (this.attSelectors.get(i) != null) {
                            trainInst = this.convToMoa
                                    .samoaInstance(this.attSelectors.get(i).reduceDimensionality(winst));
                        } else {
                            trainInst = this.convToMoa.samoaInstance(winst);
                        }

                        InstanceExample example = new InstanceExample(trainInst);
                        DoubleVector vote = new DoubleVector(this.ensemble.get(i).getVotesForInstance(trainInst));
                        this.evaluators.get(i).addResult(example, vote.getArrayRef());

                        trainInst.setWeight(trainInst.weight() * k);
                        this.ensemble.get(i).trainOnInstance(trainInst);
                    }
                }

                this.buffer = new Instances(this.getModelContext());
            }
        } catch (Exception ex) {
            Logger.getLogger(LFDD.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    @Override
    public boolean isRandomizable() {
        return true;
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        DoubleVector combinedVote = new DoubleVector();
        weka.core.Instance winst = this.convToWeka.wekaInstance(inst);

        try {
            for (int i = 0; i < this.ensemble.size(); i++) {
                DoubleVector vote;
                if (this.attSelectors.get(i) != null) {
                    this.convToMoa = new WekaToSamoaInstanceConverter();
                    vote = new DoubleVector(this.ensemble.get(i).getVotesForInstance(
                            this.convToMoa.samoaInstance(this.attSelectors.get(i).reduceDimensionality(winst))));
                } else {
                    vote = new DoubleVector(this.ensemble.get(i).getVotesForInstance(inst));
                }
                if (vote.sumOfValues() > 0.0) {
                    vote.normalize();
                    double acc = this.evaluators.get(i).getPerformanceMeasurements()[1].getValue();
                    if (acc > 0.0) {
                        vote.scaleValues(acc);
                    }
                    combinedVote.addValues(vote);
                }
            }
        } catch (Exception ex) {
            Logger.getLogger(HEFT.class.getName()).log(Level.SEVERE, null, ex);
        }

        combinedVote.normalize();

        return combinedVote.getArrayRef();
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        Measurement[] measurements = null;
        return measurements;
    }

    private AttributeSelection performFeatureSelection() {
        this.wbuffer = this.convToWeka.wekaInstances(this.buffer);
        AttributeSelection attsel = new AttributeSelection();
        SymmetricalUncertAttributeSetEval evaluator = new SymmetricalUncertAttributeSetEval();

        attsel.setEvaluator(evaluator);
        FCBFSearch search = new FCBFSearch();
        search.setThreshold(0);
        attsel.setSearch(search);

        try {
            attsel.SelectAttributes(this.wbuffer);
            attsel.selectedAttributes();
        } catch (Exception ex) {
            Logger.getLogger(HEFT.class.getName()).log(Level.SEVERE, null, ex);
        }

        return attsel;
    }
}
