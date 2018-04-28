package com.hermanbergwerf.autoSegment;

import java.io.File;
import java.io.IOException;

import net.imagej.ImageJ;

import org.scijava.widget.FileWidget;

public final class Main {
  public static void main(final String[] args) throws IOException {
    // Launch ImageJ.
    final ImageJ ij = new ImageJ();
    ij.launch(args);

    // Let the user pick an image.
    final File file = ij.ui().chooseFile(null, FileWidget.OPEN_STYLE);
    if (file == null) {
      return;
    }

    final Object data = ij.io().open(file.getAbsolutePath());
    ij.ui().show(data);

    // Run the command.
    ij.command().run(AutoSegment.class, true);
  }
}
