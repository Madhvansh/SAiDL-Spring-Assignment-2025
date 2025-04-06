<br>
In CoreML, I have introduced symmetric noise such that for the Purpose of Data Prepration, <br>
The η is random. Also, the symmetric noise function has a provision for η to be fixed like - <br>
0.6, etc. <br>
Now, for the purpose of the experiment, when the symmetric function is called, I have fixed <br>
η - 0.2, 0.4, 0.6, 0.8. So, this means that I am explicitly setting the value of eta, which <br>
means that η won't be randomized in this case and hence all the models will be trained and <br>
judged correctly. <br>