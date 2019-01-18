/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Locale;

import com.yahoo.labs.samoa.instances.ArffLoader;

import moa.classifiers.LFDD;
import moa.classifiers.bayes.NaiveBayes;
import moa.classifiers.core.driftdetection.DDM;
import moa.classifiers.core.driftdetection.EDDM;
import moa.classifiers.core.driftdetection.EWMAChartDM;
import moa.classifiers.drift.SingleClassifierDrift;
import moa.classifiers.meta.AdaptiveRandomForest;
import moa.classifiers.meta.FilteredClassifier;
import moa.classifiers.meta.HEFT;
import moa.classifiers.trees.HoeffdingAdaptiveTree;
import moa.evaluation.preview.LearningCurve;
import moa.streams.ArffFileStream;
import moa.streams.ConceptDriftStream;
import moa.streams.InstanceStream;
import moa.streams.generators.AgrawalGenerator;
import moa.streams.generators.HyperplaneGenerator;
import moa.streams.generators.LEDGeneratorDrift;
import moa.streams.generators.RandomRBFGeneratorDrift;
import moa.streams.generators.SEAFD;
import moa.streams.generators.SEAGenerator;
import moa.tasks.EvaluatePrequential;
import moa.tasks.WriteStreamToARFFFile;

public class Experiment {

    public Experiment() {
    }

    public static ArffFileStream getArffDataset(String nameFile) throws FileNotFoundException {
        ArffLoader loader = new ArffLoader(new FileReader(nameFile));
        ArffFileStream stream = new ArffFileStream(nameFile, loader.getStructure().numAttributes());
        return stream;
    }

    private ArrayList<String> mxGetDataSets() {
        ArrayList<String> namesDataSet = new ArrayList<>();

        //namesDataSet.add("data/SEAFD_A.arff");
        //namesDataSet.add("data/SEAFD_G.arff");
        //namesDataSet.add("data/SEA_A.arff");
        //namesDataSet.add("data/SEA_G.arff");
        namesDataSet.add("data/weather.arff");
        //namesDataSet.add("data/elecNormNew.arff");
        //namesDataSet.add("data/kddcup.arff");

        return namesDataSet;
    }

    public ArrayList<ClassifierTest> startProcessStream(String pathStream, int frequency, boolean save)
            throws FileNotFoundException {
        // Classifiers
        SingleClassifierDrift learnerDDM = new SingleClassifierDrift();
        learnerDDM.driftDetectionMethodOption.setCurrentObject(new DDM());
        
        FilteredClassifier learnerFilter = new FilteredClassifier();
        learnerFilter.inputStringOption.setValue("1-3");

        // Load Dataset
        ArffFileStream stream = getArffDataset(pathStream);

        ArrayList<ClassifierTest> learners = new ArrayList<>();
        // Selected algorithms
        //learners.add(new ClassifierTest(new LFDD(), "LFDD"));
        //learners.add(new ClassifierTest(new FDD(), "FDD"));
        //learners.add(new ClassifierTest(learnerDDM, "DDM"));
        learners.add(new ClassifierTest(learnerFilter, "Filtered"));
        //learners.add(new ClassifierTest(new NaiveBayes(), "NB"));
        //learners.add(new ClassifierTest(new HoeffdingAdaptiveTree(), "HAT"));
        //learners.add(new ClassifierTest(new HEFT(), "HEFT"));
        //learners.add(new ClassifierTest(new AdaptiveRandomForest(), "ARF"));

        // Prepare Learners
        for (int i = 0; i < learners.size(); i++) {
            learners.get(i).learner.setModelContext(stream.getHeader());
            learners.get(i).learner.prepareForUse();
        }

        for (int i = 0; i < learners.size(); i++) {
            String filename = Experiment.prepareFileName(learners.get(i).name, pathStream);
            // Prepare stream
            stream.prepareForUse();
            stream.restart();
            // Runs the experiment
            EvaluatePrequential evaluation = new EvaluatePrequential();
            // EvaluatePrequentialCV evaluation = new EvaluatePrequentialCV();
            evaluation.prepareForUse();
            evaluation.instanceLimitOption.setValue(100000);
            evaluation.sampleFrequencyOption.setValue(frequency);
            evaluation.dumpFileOption.setValue("./results/" + filename);
            evaluation.streamOption.setCurrentObject(stream);
            evaluation.learnerOption.setCurrentObject(learners.get(i).learner);
            LearningCurve lc = (LearningCurve) evaluation.doTask();
            // Extract information (if evaluation is Prequential, last parameter is false)
            this.getValuesForExperiment(learners.get(i), lc, false);
        }

        if (save) {
            saveFile(learners, pathStream);
        }

        return learners;
    }

    public void run(boolean save) throws IOException {
        // Prepares the folder that will contain all the results
        Experiment.prepareFolder();
        // Output File
        // PrintWriter outFile = new PrintWriter(new FileWriter("data.txt", save));

        // Prepare Datasets
        ArrayList<String> namesDataSet = this.mxGetDataSets();
        ArrayList<Double> average = new ArrayList<>();
        ArrayList<String> names = new ArrayList<>();

        for (String name : namesDataSet) {
            ArrayList<ClassifierTest> learners = this.startProcessStream(name, 1000, save);

            System.out.println("DATASET: " + name);
            System.out.printf("%12s%12s%12s%12s%12s%12s%16s\n", "Classifier", "Accuracy", "SD-Accu.", "Kappa M",
                    "kappa T", "Time", "RAM-Hours");
            System.out.println(
                    "----------------------------------------------------------------------------------------");
            for (int i = 0; i < learners.size(); i++) {
                System.out.printf("%12s %11.2f %11.2f %11.2f %11.2f %11.2f %15.9f\n", learners.get(i).name,
                        learners.get(i).accuracy, learners.get(i).sd_accuracy, learners.get(i).kappam,
                        learners.get(i).kappat, learners.get(i).time, learners.get(i).ram * Math.pow(10, 10));

                if (average.size() < learners.size()) {
                    average.add(learners.get(i).accuracy);
                    names.add(learners.get(i).name);
                } else {
                    average.set(i, average.get(i) + learners.get(i).accuracy);
                }
            }
        }

        // Final Rank
        Integer numbers[] = new Integer[average.size()];
        for (int i = 0; i < numbers.length; i++) {
            numbers[i] = i;
        }
        Arrays.sort(numbers, (final Integer o1, final Integer o2) -> Double.compare(average.get(o2), average.get(o1)));

        // Print Results
        System.out.println("\nAVERAGE RESULTS:");
        System.out.printf("%12s%12s\n", "Classifier", "Accuracy");
        System.out.println("------------------------");

        for (int i = 0; i < average.size(); i++) {
            System.out.printf("%12s %11.2f\n", names.get(numbers[i]),
                    (double) average.get(numbers[i]) / namesDataSet.size());
        }
    }

    private static void prepareFolder() {
        File folder = new File("./results/");
        File listOfFiles[];
        if (folder.exists()) {
            listOfFiles = folder.listFiles();
            for (File listOfFile : listOfFiles) {
                if (listOfFile.isFile()) {
                    // if (listOfFile.getName().endsWith(".csv")) {
                    listOfFile.delete();
                    // }
                }
            }
        } else {
            folder.mkdir();
        }
        folder = new File("/results/");
        if (folder.exists()) {
            listOfFiles = folder.listFiles();
            for (File listOfFile : listOfFiles) {
                if (listOfFile.isFile()) {
                    // if (listOfFile.getName().endsWith(".csv")) {
                    listOfFile.delete();
                    // }
                }
            }
        } else {
            folder.mkdir();
        }
    }

    private static String prepareFileName(String strClassifier, String strStream) {
        Path p = Paths.get(strStream);

        String filename = p.getFileName() + "_" + strClassifier + ".csv";
        filename = filename.trim();
        filename = filename.replace("-", "_").replace(" ", "_");
        return filename;
    }

    public void saveFile(ArrayList<ClassifierTest> learners, String name) {
        File file = new File("results/data.txt");
        NumberFormat df = NumberFormat.getCurrencyInstance(Locale.US);
        ((DecimalFormat) df).applyPattern("0.00");
        Path p = Paths.get(name);
        name = p.getFileName().toString();
        name = name.substring(0, name.lastIndexOf('.'));

        int numChunks = learners.get(0).accuracies.size();
        // Print chunck results
        try (PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter(file, true)))) {

            // Printing accuracy
            pw.printf("DataSet: %s - Accuracy\n", name);
            pw.printf("%8s,", "Instance");
            int j = 0;
            for (; j < learners.size() - 1; j++) {
                pw.printf("%8s,", learners.get(j).name);
            }
            pw.printf("%8s\n", learners.get(j).name);

            for (int i = 0; i < numChunks; i++) {
                pw.printf("%8s,", "" + learners.get(j).instances.get(i));
                for (j = 0; j < learners.size() - 1; j++) {
                    pw.printf("%8s,", df.format(learners.get(j).accuracies.get(i)));
                }
                pw.printf("%8s\n", df.format(learners.get(j).accuracies.get(i)));
            }

            // Printing Kappa M
            pw.printf("DataSet: %s - Kappa M\n", name);
            pw.printf("%8s,", "Instances");
            j = 0;
            for (; j < learners.size() - 1; j++) {
                pw.printf("%8s,", learners.get(j).name);
            }
            pw.printf("%8s\n", learners.get(j).name);

            for (int i = 0; i < numChunks; i++) {
                pw.printf("%8s,", "" + learners.get(j).instances.get(i));
                for (j = 0; j < learners.size() - 1; j++) {
                    pw.printf("%8s,", df.format(learners.get(j).kappams.get(i)));
                }
                pw.printf("%8s\n", df.format(learners.get(j).kappams.get(i)));
            }

            // Printing Kappa T
            pw.printf("DataSet: %s - Kappa T\n", name);
            pw.printf("%8s,", "Instances");
            j = 0;
            for (; j < learners.size() - 1; j++) {
                pw.printf("%8s,", learners.get(j).name);
            }
            pw.printf("%8s\n", learners.get(j).name);

            for (int i = 0; i < numChunks; i++) {
                pw.printf("%8s,", "" + learners.get(j).instances.get(i));
                for (j = 0; j < learners.size() - 1; j++) {
                    pw.printf("%8s,", df.format(learners.get(j).kappats.get(i)));
                }
                pw.printf("%8s\n", df.format(learners.get(j).kappats.get(i)));
            }

            // Printing Average Results
            PrintWriter pwAcc = new PrintWriter(
                    new BufferedWriter(new FileWriter(new File("results/accuracy.txt"), true)));
            PrintWriter pwKam = new PrintWriter(
                    new BufferedWriter(new FileWriter(new File("results/kappam.txt"), true)));
            PrintWriter pwKat = new PrintWriter(
                    new BufferedWriter(new FileWriter(new File("results/kappat.txt"), true)));
            PrintWriter pwTim = new PrintWriter(new BufferedWriter(new FileWriter(new File("results/time.txt"), true)));

            pwAcc.printf("%15s", name);
            for (j = 0; j < learners.size() - 1; j++) {
                pwAcc.printf("%8s,", df.format(learners.get(j).accuracy));
            }
            pwAcc.printf("%8s\n", df.format(learners.get(j).accuracy));

            pwKam.printf("%12s", name);
            for (j = 0; j < learners.size() - 1; j++) {
                pwKam.printf("%8s,", df.format(learners.get(j).kappam));
            }
            pwKam.printf("%8s\n", df.format(learners.get(j).kappam));

            pwKat.printf("%12s", name);
            for (j = 0; j < learners.size() - 1; j++) {
                pwKat.printf("%8s,", df.format(learners.get(j).kappat));
            }
            pwKat.printf("%8s\n", df.format(learners.get(j).kappat));

            pwTim.printf("%12s", name);
            for (j = 0; j < learners.size() - 1; j++) {
                pwTim.printf("%8s,", df.format(learners.get(j).time));
            }
            pwTim.printf("%8s\n", df.format(learners.get(j).time));

            pwAcc.close();
            pwKam.close();
            pwKat.close();
            pwTim.close();

        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    private void getValuesForExperiment(ClassifierTest classifier, LearningCurve lc, boolean isCV) {
        int indexAcc = -1;
        int indexKappam = -1;
        int indexKappat = -1;
        int indexCpuTime = -1;
        int indexRamHours = -1;
        int indexInstances = -1;

        int index = 0;
        for (String s : lc.headerToString().split(",")) {
            if (s.contains("[avg] classifications correct") || (!isCV && (s.contains("classifications correct")))) {
                indexAcc = index;
            } else if (s.contains("time")) {
                indexCpuTime = index;
            } else if (s.contains("RAM-Hours")) {
                indexRamHours = index;
            } else if (s.contains("[avg] Kappa M") || (!isCV && (s.contains("Kappa M")))) {
                indexKappam = index;
            } else if (s.contains("[avg] Kappa Temporal") || (!isCV && (s.contains("Kappa Temporal")))) {
                indexKappat = index;
            } else if (s.contains("learning evaluation instances")) {
                indexInstances = index;
            }
            index++;
        }

        // Reading all values
        for (int entry = 0; entry < lc.numEntries(); entry++) {

            classifier.accuracies.add(lc.getMeasurement(entry, indexAcc));
            classifier.kappams.add(lc.getMeasurement(entry, indexKappam));
            classifier.kappats.add(lc.getMeasurement(entry, indexKappat));
            classifier.instances.add(lc.getMeasurement(entry, indexInstances));
        }
        // Calculating statistical values
        classifier.mxCalculateValues();
        // but both cpu time and ram hours are only the final values obtained
        // since they represent the processing of the entire stream
        classifier.time = lc.getMeasurement(lc.numEntries() - 1, indexCpuTime);
        classifier.ram = lc.getMeasurement(lc.numEntries() - 1, indexRamHours);
    }

    public void generateSEAFDStream(String name, int width) {
        SEAFD concept1 = new SEAFD();
        concept1.balanceClassesOption.set();
        concept1.numRandomAttsOption.setValue(10);
        concept1.prepareForUse();

        SEAFD concept2 = new SEAFD();
        concept2.instanceRandomSeedOption.setValue(813727813);
        concept2.balanceClassesOption.set();
        concept2.numRandomAttsOption.setValue(10);
        concept2.prepareForUse();

        SEAFD concept3 = new SEAFD();
        concept3.instanceRandomSeedOption.setValue(4786123);
        concept3.balanceClassesOption.set();
        concept3.numRandomAttsOption.setValue(10);
        concept3.prepareForUse();

        SEAFD concept4 = new SEAFD();
        concept4.instanceRandomSeedOption.setValue(5266498);
        concept4.balanceClassesOption.set();
        concept4.numRandomAttsOption.setValue(10);
        concept4.prepareForUse();

        // Creates the final stream
        ArrayList<InstanceStream> concepts = new ArrayList<>();
        concepts.add(concept1);
        concepts.add(concept2);
        concepts.add(concept3);
        concepts.add(concept4);
        int instStream = 25000;

        WriteStreamToARFFFile saver = new WriteStreamToARFFFile();
        saver.prepareForUse();
        saver.streamOption.setCurrentObject(getJoinStreams(concepts, width, instStream));
        saver.maxInstancesOption.setValue(100000);
        saver.arffFileOption.setValue("./data/" + name + ".arff");
        saver.doTask();
    }

    public void generateSEAStream(String name, int width) {
        SEAGenerator concept1 = new SEAGenerator();
        concept1.balanceClassesOption.set();
        concept1.functionOption.setValue(1);
        concept1.prepareForUse();

        SEAGenerator concept2 = new SEAGenerator();
        concept2.instanceRandomSeedOption.setValue(813727813);
        concept2.balanceClassesOption.set();
        concept2.functionOption.setValue(2);
        concept2.prepareForUse();

        SEAGenerator concept3 = new SEAGenerator();
        concept3.instanceRandomSeedOption.setValue(4786123);
        concept3.balanceClassesOption.set();
        concept3.functionOption.setValue(3);
        concept3.prepareForUse();

        SEAGenerator concept4 = new SEAGenerator();
        concept4.instanceRandomSeedOption.setValue(5266498);
        concept4.balanceClassesOption.set();
        concept4.functionOption.setValue(4);
        concept4.prepareForUse();
        
        // Creates the final stream
        ArrayList<InstanceStream> concepts = new ArrayList<>();
        concepts.add(concept1);
        concepts.add(concept2);
        concepts.add(concept3);
        concepts.add(concept4);
        int instStream = 25000;

        WriteStreamToARFFFile saver = new WriteStreamToARFFFile();
        saver.prepareForUse();
        saver.streamOption.setCurrentObject(getJoinStreams(concepts, width, instStream));
        saver.maxInstancesOption.setValue(100000);
        saver.arffFileOption.setValue("./data/" + name + ".arff");
        saver.doTask();
    }

    public void generateAGRAWALStream(String name, int width) {
        AgrawalGenerator concept1 = new AgrawalGenerator();
        concept1.balanceClassesOption.set();
        concept1.prepareForUse();

        AgrawalGenerator concept2 = new AgrawalGenerator();
        concept2.balanceClassesOption.set();
        concept2.functionOption.setValue(2);
        concept2.prepareForUse();

        AgrawalGenerator concept3 = new AgrawalGenerator();
        concept3.balanceClassesOption.set();
        concept3.functionOption.setValue(3);
        concept3.prepareForUse();

        AgrawalGenerator concept4 = new AgrawalGenerator();
        concept4.balanceClassesOption.set();
        concept4.functionOption.setValue(4);
        concept4.prepareForUse();

        // Creates the final stream
        ArrayList<InstanceStream> concepts = new ArrayList<>();
        concepts.add(concept1);
        concepts.add(concept2);
        concepts.add(concept3);
        concepts.add(concept4);
        int instStream = 25000;

        WriteStreamToARFFFile saver = new WriteStreamToARFFFile();
        saver.prepareForUse();
        saver.streamOption.setCurrentObject(getJoinStreams(concepts, width, instStream));
        saver.maxInstancesOption.setValue(instStream * 4);
        saver.arffFileOption.setValue("./data/" + name + ".arff");
        saver.doTask();
    }

    public void generateLEDStream(String name, int width) {
        LEDGeneratorDrift concept1 = new LEDGeneratorDrift();
        concept1.numberAttributesDriftOption.setValue(1);
        concept1.prepareForUse();

        LEDGeneratorDrift concept2 = new LEDGeneratorDrift();
        concept2.numberAttributesDriftOption.setValue(3);
        concept2.prepareForUse();

        LEDGeneratorDrift concept3 = new LEDGeneratorDrift();
        concept3.numberAttributesDriftOption.setValue(5);
        concept3.prepareForUse();

        LEDGeneratorDrift concept4 = new LEDGeneratorDrift();
        concept4.numberAttributesDriftOption.setValue(7);
        concept4.prepareForUse();

        // Creates the final stream
        ArrayList<InstanceStream> concepts = new ArrayList<>();
        concepts.add(concept1);
        concepts.add(concept2);
        concepts.add(concept3);
        concepts.add(concept4);
        int instStream = 25000;

        WriteStreamToARFFFile saver = new WriteStreamToARFFFile();
        saver.prepareForUse();
        saver.streamOption.setCurrentObject(getJoinStreams(concepts, width, instStream));
        saver.maxInstancesOption.setValue(instStream * 4);
        saver.arffFileOption.setValue("./data/" + name + ".arff");
        saver.doTask();
    }

    public void generateHyperplaneStream(String name) {
        HyperplaneGenerator concept1 = new HyperplaneGenerator();
        concept1.numDriftAttsOption.setValue(10);
        concept1.magChangeOption.setValue(0.001);
        concept1.prepareForUse();

        WriteStreamToARFFFile saver = new WriteStreamToARFFFile();
        saver.prepareForUse();
        saver.streamOption.setCurrentObject(concept1);
        saver.maxInstancesOption.setValue(100000);
        saver.arffFileOption.setValue("./data/" + name + ".arff");
        saver.doTask();
    }

    public void generateRBFDriftStream(String name, double speedChange) {
        RandomRBFGeneratorDrift concept1 = new RandomRBFGeneratorDrift();
        concept1.numClassesOption.setValue(5);
        concept1.speedChangeOption.setValue(speedChange);
        concept1.prepareForUse();

        WriteStreamToARFFFile saver = new WriteStreamToARFFFile();
        saver.prepareForUse();
        saver.streamOption.setCurrentObject(concept1);
        saver.maxInstancesOption.setValue(100000);
        saver.arffFileOption.setValue("./data/" + name + ".arff");
        saver.doTask();
    }

    ConceptDriftStream getJoinStreams(ArrayList<InstanceStream> concepts, int width, int instStream){
        int numConcepts = concepts.size();
        ConceptDriftStream strTmp;

        ConceptDriftStream str =  new ConceptDriftStream();
        str.streamOption.setCurrentObject(concepts.get(numConcepts - 2));
        str.positionOption.setValue(instStream);
        str.widthOption.setValue(width);
        str.driftstreamOption.setCurrentObject(concepts.get(numConcepts - 1));
        str.prepareForUse();

        for (int i = numConcepts - 3; i >= 0; i--) {
            strTmp = new ConceptDriftStream(); 
            strTmp = (ConceptDriftStream) str.copy();
            str = new ConceptDriftStream();
            str.streamOption.setCurrentObject(concepts.get(i));
            str.positionOption.setValue(instStream);
            str.widthOption.setValue(width);
            str.driftstreamOption.setCurrentObject(strTmp);
            str.prepareForUse();
        }

        return str;
    }

    void generateStreams(){        
        /*this.generateAGRAWALStream("AGR_G", 5000);
        this.generateAGRAWALStream("AGR_A", 1);*/
        this.generateSEAStream("SEA_G", 5000);
        this.generateSEAStream("SEA_A", 1);
        this.generateSEAFDStream("SEAFD_G", 5000);
        this.generateSEAFDStream("SEAFD_A", 1);
        /*this.generateLEDStream("LED_G", 5000);
        this.generateLEDStream("LED_A", 1);
        this.generateRBFDriftStream("LED_M", 0.0001);
        this.generateRBFDriftStream("LED_F", 0.001);
        this.generateHyperplaneStream("HYPER");*/                     
    }

    public static void main(String[] args) throws IOException {
        Experiment exp = new Experiment();
        exp.run(true);
        //exp.generateStreams();
    }
}