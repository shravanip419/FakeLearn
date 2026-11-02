BDA PRACTICAL EXAM PACK — EXPERIMENTS 1 TO 7
============================================

IMPORTANT
---------
• Experiment 8 (mini project) is intentionally excluded.
• Each experiment has its own folder.
• Each folder contains INSTRUCTIONS.txt and separate source/code files.
• These instructions target Ubuntu/Debian-style Linux terminals used in college labs.
• Run commands in the terminal type stated in the instructions. Do not paste Linux commands into MySQL, mongosh, Pig Grunt, or R.
• Replace paths, passwords, table names, and dataset filenames only when your lab PC uses different values.
• No one can guarantee zero errors on an unknown college PC: installed versions, permissions, internet access, and missing datasets can differ. The first step in every experiment is a version/availability check. If a required tool is already installed, do not reinstall it.

EXPERIMENTS
-----------
Experiment_1_Hadoop_Pseudo_Distributed
Experiment_2_Sqoop_MySQL
Experiment_3_MapReduce
Experiment_4_MongoDB
Experiment_5_Apache_Pig
Experiment_6_Bloom_Filter
Experiment_7_R_Data_Analysis

BEFORE THE PRACTICAL
--------------------
1. Extract this ZIP (do not run files from inside the ZIP preview).
2. Copy the extracted folder to your Linux home directory or Desktop.
3. Open Terminal and use: cd ~/BDA_Practical_7_Experiments
4. Read the relevant experiment's INSTRUCTIONS.txt from top to bottom.
5. Keep your college-provided datasets available. Experiment 5 needs crop_production.csv and the Amazon CSV. If the Amazon dataset's columns differ, edit the LOAD schema in the .pig file.
6. Experiment 2 needs MySQL server, Sqoop, Hadoop/HDFS, and a MySQL JDBC connector. Sqoop is an old/retired tool; compatibility with the installed Hadoop/Java versions is environment-dependent. Use the college lab's tested installation if available.
7. Experiment 1's Hadoop setup is persistent. Do NOT format the NameNode repeatedly. Formatting an existing NameNode can destroy HDFS metadata.
