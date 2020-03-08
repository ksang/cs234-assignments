# cs234-assignments
Stanford CS234: Reinforcement Learning assignments and practices

### Overview

- This project are assignment solutions and practices of Stanford class CS234.

- The assignments are for Winter 2020, video recordings are available on [Youtube](https://www.youtube.com/playlist?list=PLoROMvodv4rOSOPzutgyCTapiGlY2Nd8uv).

- For detailed information of the class, goto: [CS234 Home Page](https://web.stanford.edu/class/cs234/)

- Assignments will be updated with my solutions, currently *WIP*.

### Index

There are totally three assignments, each of them has programming part and written part.

[Assignment 1 written](/assignment1_written)

- Grid World

- Value of Different Policies

- Fixed Point

[Assignment 1 coding](/assignment1_coding)

- Frozen Lake MDP, policy evaluation, policy improvement, policy iteration, value iteration.

- Details see source code: [vi_and_pi.py](/assignment1/vi_and_pi.py)

[Assignment 2 written](/assignment2_written)

[Assignment 2 coding](/assignment2_coding)

- Q-learning

- Linear Approximation

- Implementing DeepMind's DQN

- DQN on Atari

- n-step Estimators


### Notes on minimum LaTex environment installation on OSX

- Install basic tex package

        brew cask install basictex

- Install missing packages from the assignments.

    * if some package is missing, tex compiler such as `pdftex` will give you their name, e.g. `nicefrac.sty`.
    * Search the package name on [CTAN](https://www.ctan.org/), and get the parent package name, e.g. `units`
    * Below command will install the package:

            sudo tlmgr install units

    * I already did this for you, here is the command for install all dependencies for this assignment:

            sudo tlmgr update --self
            sudo tlmgr install units fullpage preprint \
                                wrapfig was apptools appendix \
                                titlesec enumitem breakurl \
                                algorithm2e ifoddpage relsize cm-super \
                                lastpage comment framed biblatex typewriter

- Use your favorite editor with LaTeX support and enjoy the math. I'm using [Atom](https://atom.io/) with [LaTex](https://atom.io/packages/latex) and [pdf-view](https://atom.io/packages/pdf-view) package.

Any questions or advice, just open an issue or pull request.
