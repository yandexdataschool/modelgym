Installation
============
To build documentation locally you need to install::

    pip3 install -r requirements-dev.txt

To update examples in documentation with ``generate_examples_rst.sh`` you need to install `pandoc <https://pandoc.org/installing.html>`_.

Build documentation
===================
To build documentation locally run from ``docs`` directory::

    make html

And then you can open documentation in browser at ``file://full_path_to_docs/build/html/index.html``.

To generate rst examples from ipynb simply run ``generate_examples_rst.sh`` script.
If you are going to add new example you should also add it into ``index.rst``.
