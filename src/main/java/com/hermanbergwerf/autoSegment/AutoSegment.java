package com.hermanbergwerf.autoSegment;

import net.imagej.Dataset;
import net.imagej.tensorflow.TensorFlowService;
import net.imglib2.img.Img;
import net.imglib2.type.numeric.real.FloatType;

import org.scijava.ItemIO;
import org.scijava.command.Command;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

import org.tensorflow.TensorFlow;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session.Runner;

@Plugin(type = Command.class, menuPath = "Plugins>Segmentation>Auto segment")
public class AutoSegment implements Command {
  private static final String MODEL_LOCATION = "./examples/cells/tf_model";
  private static final String MODEL_TAG = "unet_cells";

  @Parameter
  private TensorFlowService tensorFlowService;

  @Parameter
  private LogService log;

  @Parameter
  private Dataset inputImage;

  @Parameter(type = ItemIO.OUTPUT)
  private Img<FloatType> outputImage;

  @Override
  public void run() {
    try {
      log.info("TensorFlow version: " + TensorFlow.version());
      final SavedModelBundle model = SavedModelBundle.load(MODEL_LOCATION, MODEL_TAG);
      final Runner runner = model.session().runner();
      log.info("Loaded selected model.");
    } catch (final Exception exc) {
      log.error(exc);
    }
  }
}
