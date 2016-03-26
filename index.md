
<div style="padding: 20px; margin: 0 auto; width:60%; text-align:center" > 
    <h1 style="font-weight: normal; font-size: 20pt;"> 
	    Bottom-Up and Top-Down Reasoning with Hierarchical Rectified Gaussians 
	</h1>

	<table align="center" width="30%" style="font-size: 14pt">
	    <tr>
	        <td> <a href="http://www.ics.uci.edu/~peiyunh/">Peiyun Hu</a> </td>
	        <td> <a href="http://www.cs.cmu.edu/~deva/">Deva Ramanan</a> </td>
	    </tr>
	</table>
	
	<div style="padding: 10px;">
	    <a href="./teaser.png"> <img width=50% src="./teaser.png"/> </a>
	</div>

    <h3 style="font-weight: normal; font-size: 15pt;"> Abstract </h3>
	<div style="margin: 0 auto; width: 60%; text-align: left"> 
	Convolutional neural nets (CNNs) have demonstrated remarkable performance in
	recent history. Such approaches tend to work in a “unidirectional” bottom-up
	feed-forward fashion. However, practical experience and biological evidence
	tells us that feedback plays a crucial role, particularly for detailed
	spatial understanding tasks. This work explores “bidirectional”
	architectures that also reason with top-down feedback: neural units are
	influenced by both lower and higher-level units.
	<br> <br> We do so by treating units as rectified latent variables in a
	quadratic energy function, which can be seen as a hierarchical Rectified
	Gaussian model (RGs). We show that RGs can be optimized with a
	quadratic program (QP), that can in turn be optimized with a recurrent
	neural network (with rectified linear units). This allows RGs to be trained
	with GPU-optimized gradient descent. From a theoretical perspective, RGs
	help establish a connection between CNNs and hierarchical probabilistic
	models. From a practical perspective, RGs are well suited for detailed
	spatial tasks that can benefit from top-down reasoning. We illustrate them
	on the challenging task of keypoint localization under occlusions, where
	local bottom-up evidence may be misleading. We demonstrate state-of-the-art
	results on challenging benchmarks.
	</div>
	
    <h3 style="font-weight: normal; font-size: 15pt;"> Paper </h3>
	<div style="margin: 0 auto; width: 60%; text-align: center;"> 
		CVPR 2016: <font color="red"> coming soon </font>
	</div>

    <h3 style="font-weight: normal; font-size: 15pt;"> Code </h3>
	<div style="margin: 0 auto; width: 60%; text-align: center"> 	
		See <a href="https://github.com/peiyunh/rg-mpii/">Github</a> for our
        code and trained models.
	</div>
	
</div>
