package com.hermanbergwerf.autoSegment;

import java.io.IOException;
import java.util.Arrays;

import net.imagej.tensorflow.TensorFlowService;
import net.imagej.tensorflow.Tensors;

import net.imglib2.img.Img;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.converter.Converter;
import net.imglib2.converter.Converters;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;

import org.scijava.command.Command;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session.Runner;

@Plugin(type = Command.class, menuPath = "Plugins>Segmentation>Auto segment")
public class AutoSegment<T extends RealType<T>> implements Command {
  private static final String MODEL_LOCATION = "./examples/cells/tf_model";
  private static final String MODEL_TAG = "unet_cells";

  @Parameter
  private TensorFlowService tensorFlowService;

  @Parameter
  private LogService log;

  @Parameter(label = "Input image")
  private Img<T> inImage;

  @Override
  public void run() {
    try {
      // Validate input.
      validateFormat(inImage);

      // Load Model.
      log.info("TensorFlow version: " + TensorFlow.version());
      final SavedModelBundle model = SavedModelBundle.load(MODEL_LOCATION, MODEL_TAG);
      final Runner runner = model.session().runner();
      log.info("Loaded selected model.");

      // Convert current image the the correct format: single channel floating
      // point image in the range [0.0, 1.0].
      // Normalize with imagej library rather than tensorflow because it happens
      // to be relatively convenient.
      RandomAccessibleInterval<FloatType> normalizedImage = normalize(inImage);

      // Split image into patches, for each patch run a prediction, and
      // threshold the cell feature (channel 2) at .8 to get a binary mask.
      // All binary masks are collected in one complete binary mask.
      @SuppressWarnings("unchecked") // Because Tensors is not well implemented.
      final Tensor<Float> inTensor = Tensors.tensor(normalizedImage);
      log.info(String.format("Input shape: %s", Arrays.toString(inTensor.shape())));

      // Generate regions from the binary mask and add them to the ROI manager
      // of ImageJ.
      // TODO: Try to do patching and stitching in Python rather than in Java
      // since all Java API's here are 'limited' (and bad docs).
    } catch (final Exception exc) {
      log.error(exc);
    }
  }

  /// Check if the opened image has a supported file type.
  private void validateFormat(final Img<T> image) throws IOException {
    final int ndims = image.numDimensions();
    if (ndims != 2) {
      final long[] dims = new long[ndims];
      image.dimensions(dims);
      throw new IOException(//
          String.format("Can only process 2D images, not an image with " + //
              "%d dimensions (%s)", ndims, Arrays.toString(dims)));
    }
  }

  /// Normalize to [0.0, 1.0]. Copied from MicroscopeImageFocusQualityClassifier
  /// where this is for some reason theoretically unsound.
  private RandomAccessibleInterval<FloatType> normalize(//
      final RandomAccessibleInterval<T> image) {
    final double min = image.randomAccess().get().getMinValue();
    final double max = image.randomAccess().get().getMaxValue();
    Converter<T, FloatType> normalizer = (input, output) -> //
    output.setReal((input.getRealDouble() - min) / (max - min));
    return Converters.convert(image, normalizer, new FloatType());
  }
}
