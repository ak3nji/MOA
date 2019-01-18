/*
 *    DriftDetectionMethodClassifier.java
 *    Copyright (C) 2008 University of Waikato, Hamilton, New Zealand
 *    @author Manuel Baena (mbaena@lcc.uma.es)
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program. If not, see <http://www.gnu.org/licenses/>.
 */
package moa.classifiers;

import java.util.logging.Level;
import java.util.logging.Logger;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.SamoaToWekaInstanceConverter;
import com.yahoo.labs.samoa.instances.WekaToSamoaInstanceConverter;

import moa.classifiers.core.driftdetection.ChangeDetector;
import moa.core.Measurement;
import moa.options.ClassOption;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.FCBFSearch;
import weka.attributeSelection.SymmetricalUncertAttributeSetEval;

/**
 * Class for handling concept drift datasets with a wrapper on a classifier.
 * <p>
 *
 * Valid options are:
 * <p>
 *
 * -l classname <br>
 * Specify the full class name of a classifier as the basis for the concept
 * drift classifier.
 * <p>
 * -d Drift detection method to use<br>
 *
 * @author Manuel Baena (mbaena@lcc.uma.es)
 * @version 1.1
 */
public class FDD extends AbstractClassifier implements MultiClassClassifier {

    private static final long serialVersionUID = 1L;

    @Override
    public String getPurposeString() {
        return "Classifier that replaces the current classifier with a new one when a change is detected in accuracy.";
    }

    public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l', "Classifier to train.", Classifier.class,
            "bayes.NaiveBayes");

    public ClassOption driftDetectionMethodOption = new ClassOption("driftDetectionMethod", 'd',
            "Drift detection method to use.", ChangeDetector.class, "DDM");

    protected Classifier classifier;
    protected Instances buffer;
    protected AttributeSelection attSelector;
    protected ChangeDetector driftDetectionMethod;
    protected boolean newClassifierReset;
    protected boolean completeBuffer;

    protected weka.core.Instances wbuffer;
    protected SamoaToWekaInstanceConverter convToWeka;
    protected WekaToSamoaInstanceConverter convToMoa;

    protected int ddmLevel;
    protected long minSize, numInstances;

    public static final int DDM_INCONTROL_LEVEL = 0;
    public static final int DDM_WARNING_LEVEL = 1;
    public static final int DDM_OUTCONTROL_LEVEL = 2;

    @Override
    public void resetLearningImpl() {
        this.classifier = ((Classifier) getPreparedClassOption(this.baseLearnerOption)).copy();
        this.classifier.resetLearning();
        this.driftDetectionMethod = ((ChangeDetector) getPreparedClassOption(this.driftDetectionMethodOption)).copy();
        this.newClassifierReset = false;
        this.buffer = null;
        this.completeBuffer = false;
        this.minSize = 100;
        this.convToWeka = new SamoaToWekaInstanceConverter();
    }

    protected int changeDetected = 0;
    protected int warningDetected = 0;

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        this.numInstances++;

        if (this.buffer == null) {
            this.buffer = new Instances(inst.dataset());
        }

        Instance trainInst = inst.copy();

        if (this.attSelector != null) {
            weka.core.Instance winst = this.convToWeka.wekaInstance(inst);
            try {
                trainInst = this.convToMoa.samoaInstance(this.attSelector.reduceDimensionality(winst));
            } catch (Exception ex) {
                Logger.getLogger(LFDD.class.getName()).log(Level.SEVERE, null, ex);
            }
        }

        // It is necessary complete the buffer
        if (this.completeBuffer) {
            if (this.buffer.numInstances() < this.minSize) {
                this.buffer.add(inst);
            } else {
                this.retrainClassifier();
            }
        } else {
            this.driftDetectionMethod.input(this.classifier.correctlyClassifies(trainInst) ? 0.0 : 1.0);
            this.ddmLevel = DDM_INCONTROL_LEVEL;

            if (this.driftDetectionMethod.getChange()) {
                this.ddmLevel = DDM_OUTCONTROL_LEVEL;
            }
            if (this.driftDetectionMethod.getWarningZone()) {
                this.ddmLevel = DDM_WARNING_LEVEL;
            }

            switch (this.ddmLevel) {
            case DDM_INCONTROL_LEVEL:
                this.newClassifierReset = true;
                break;

            case DDM_WARNING_LEVEL:
                if (this.newClassifierReset == true) {
                    this.warningDetected++;
                    this.newClassifierReset = false;
                    this.buffer = new Instances(this.getModelContext());
                }
                this.buffer.add(inst);
                break;

            case DDM_OUTCONTROL_LEVEL:
                this.changeDetected++;
                // Buffer is not full
                if (this.buffer.numInstances() < this.minSize) {
                    this.completeBuffer = true;
                } else {
                    this.retrainClassifier();
                }
                break;

            default:
                // System.out.println("ERROR!");
                break;
            }
        }

        this.classifier.trainOnInstance(trainInst);
    }

    public double[] getVotesForInstance(Instance inst) {
        Instance trainInst = inst.copy();

        try {
            if (this.attSelector != null) {
                weka.core.Instance winst = this.convToWeka.wekaInstance(inst);
                trainInst = this.convToMoa.samoaInstance(this.attSelector.reduceDimensionality(winst));
            }
        } catch (Exception ex) {
            Logger.getLogger(LFDD.class.getName()).log(Level.SEVERE, null, ex);
        }
        return this.classifier.getVotesForInstance(trainInst);
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
        Measurement[] measurements = null;
        return measurements;
    }

    private void retrainClassifier() {
        Instance trainInst = null;
        // Feature Selection performed over data chunck
        this.attSelector = this.performFeatureSelection();
        this.classifier = null;
        this.classifier = ((Classifier) getPreparedClassOption(this.baseLearnerOption)).copy();
        this.classifier.resetLearning();
        this.driftDetectionMethod = ((ChangeDetector) getPreparedClassOption(this.driftDetectionMethodOption)).copy();
        this.convToMoa = new WekaToSamoaInstanceConverter();
        this.completeBuffer = false;

        try {
            for (weka.core.Instance winst : this.wbuffer) {
                // All attributes?
                if (this.attSelector != null) {
                    trainInst = this.convToMoa.samoaInstance(this.attSelector.reduceDimensionality(winst));
                } else {
                    trainInst = this.convToMoa.samoaInstance(winst);
                }
                this.classifier.trainOnInstance(trainInst);
            }
        } catch (Exception ex) {
            Logger.getLogger(FDD.class.getName()).log(Level.SEVERE, null, ex);
        }

        this.buffer = new Instances(this.getModelContext());
    }

    private AttributeSelection performFeatureSelection() {
        System.out.println(this.buffer.numInstances() + " - " + this.numInstances);
        this.wbuffer = this.convToWeka.wekaInstances(this.buffer);
        AttributeSelection attsel = new AttributeSelection();
        SymmetricalUncertAttributeSetEval evaluator = new SymmetricalUncertAttributeSetEval();

        attsel.setEvaluator(evaluator);
        FCBFSearch search = new FCBFSearch();
        search.setThreshold(0);
        attsel.setSearch(search);

        try {
            attsel.SelectAttributes(this.wbuffer);
            System.out.println(attsel.selectedAttributes().length);
        } catch (Exception ex) {
            Logger.getLogger(LFDD.class.getName()).log(Level.SEVERE, null, ex);
        }

        return attsel;
    }
}