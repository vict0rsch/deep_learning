# MOVING

I'm moving the tutorial to **[vict0rsch.github.io](http://vict0rsch.github.io)** to make it nicer for you guys. This is an ongoing process and your feedbacks are welcome! Basically it will be betetr organized and there will be comments possible. This repo will stay active for the code, issues, pull requests and collaborations.

---



*Disclaimer : I am not an expert, this is only a small contribution from a self(internet)-taught beginner with AWS who wants to synthesize what he's learned. And I'm happy to learn even more if this is incorrect or (surely) improvable*

-----

# Launch your Amazon GPU instance

The purpose here is to get you to launch an Amazon instance from an AMI.

What the hell is an AMI? It is an Amazon Machine Image, which basically describes the software installed on a machine. So here we are going to launch a pre-configured instance. 

It will (mainly) have :

* Ubuntu Server 16.04 as OS
* [Anaconda](https://www.continuum.io/) 4.2.0 (scientific Python distribution)
* Python 3.5
* Cuda 8.0 ("parallel computing platform and programming model", used to send code to the GPU)
* cuDNN 5.1 (Cuda's library for Deep Learning used by Tensorflow and Theano)
* [Tensorflow](https://www.tensorflow.org/) 0.12 for Python 3.5 and GPU-enabled
* [Keras](https://keras.io/) 1.1.2 (use with Tensorflow backend)

*To be done:*
* [Theano](http://deeplearning.net/software/theano/index.html) 0.8 I currently have issues during `theano.test()` if anyone wants to help
* [Lasagne](http://lasagne.readthedocs.io/en/latest/index.html) 0.1
* rsub by default

This AMI can be seen as a list of softwares, it does not specify the hardware you are going to use. Therefore the 3 main steps of this tutorial will be :

1. Select the AMI
2. Select the Instance (hardware)
3. Connect to it

## Before you go

If you have never launched an EC2 instance (maybe if you're changing region?) your EC2 instances limit is set to 0 by default. Therefore at launch time, at the end of the whole process you'll be prompted a **"Launch Failed" error...**

To save you some time here is what you've got to do : go to http://aws.amazon.com/contact-us/ec2-request and write them a nice message asking for a higher limit. I randomly asked for 3 and got 5. Maybe it's standard. Don't forget to select the right region as AMIs are region-specific. So is pricing. 

I asked for a "Web" contact method and they got back to me in **2 business days**. So if you're in a hurry maybe the phone method is faster. I don't know. People with experience could elaborate on that. 

___
## 0. In a nutshell

1. Go to EC2 instances in North California region
2. Select `vict0rsch-1.0` community AMI
3. Select `g2.2xlarge` instance
4. Set up Security group
5. Download RSA private key
6. Connect *via* ssh : `ssh -i key.pem ubuntu@address`

___

## 1. Selecting an AMI

#### Region

I assume you have created an AWS account. The AMI we are going to use (mine, feel free to comment on this) is located in the **North California** region. So be sure that once you get to the AWS Console you select North California in the top right corner.

#### AMI

Now in "Services" (top left) select EC2 and click that big beautiful blue button that says "Launch Instance".

You'll be asked to "Choose an Amazon Machine Image (AMI)". On the left, click "**Community AMIs**", look for `vict0rsch-1.0` and select it.

## 2. Lauching the instance

#### Choose an Instance Type

To speed up computation, we'll use a GPU instance, so select a `g2.2xlarge` instance. Now click on "Review and launch"

#### Security groups

We need to define who's gonna be able to connect to the instance (by default it can connect itself to any address via any protocol). 

So create a new security group with a name and description of your choice.

In "Source" click on **My IP**, and the line should look like `ssh | TCP | 22 | My IP your.ip/32`

#### Storage

The whole setup takes a bit more than 9GB. By default the AMI makes the instance have 24GB storage. See the "Storage" tab (no shit...) to edit this according to your needs.

Now "**Launch**" !

#### Key Pair

You've been prompted with a Key pair choice. This is about the **private RSA key** that will allow you to connect to the instance. Create a new one, give it a name and download it. Then move it wherever you want but it might be a good idea not to let it in your downloads directory.

You can create a new key at every instance launch but you will need the declared one to connect to the instance. 

OK! Your instance is being launched by Amazon, click on the blue link with the instance ID or go to your console (it's the same anyway) and go in the "**Instances**" tab. It will only take a few minutes before it's running and you can connect to it.

## 3. Connecting to the instance

#### SSH

You'll connect to the instance *via* ssh. To do so you need the instance's address which is found writen in bold at the bottom of the window when you select the instance. It looks like `Public DNS: ec2-54-183-195-215.us-west-1.compute.amazonaws.com`

The default user of Ubuntu Server instances is `ubuntu` but on other AMIs such as the Amazon Linux ones it is usually `ec2-user`.

We'll use te flag `-i` to specify that we use an identity file i.e. the key we downloaded from Amazon.

**Now connect to your instance :**

`ssh -i path_to_key/key.pem ubuntu@ec2-54-183-195-215.us-west-1.compute.amazonaws.com`

(of course don't use this address, rather your `Public DNS` above)

Accept to add the RSA key and there you are! You should be prompted with something like `ubuntu@ip-172-31-5-202:~$`

#### SCP

To **copy** (transfer) files from your computer to the instance use `scp` as follows : 

`scp -i path_to_key/key.pem file_on_your_computer ubuntu@ec2-54-183-195-215.us-west-1.compute.amazonaws.com:path_on_remote_instance`

Add the flags `-rp` to transfer directories.

#### SSH - SCP Errors

If you are prompted with the  `Permission denied (publickey)`**error** or the operation is timed out it can mean:

* Your instance is not running (yet?)
* That you are not using the right RSA key
* You are trying to log into another machine (wrong address)
* You are trying to log in with the wrong user 
* Your instance does not allow your IP (so go and add it in its security group)

I also had `Connection closed by <address>` errors. So far the only way I found was to create a new KeyPair from the Amazon console. You can save your progress by taking a snapshot, terminating the instance and starting a new one with a new KeyPair from the snapshot.

If the `scp` command does not say anything and fails, check that you did not forget the path on the remote host at the end of the address : `[...].com:~/` for instance.

Anyway you can have `scp` and `ssh` be more verbose using `-v` `-vv` or `-vvv` depending on the details you want.

#### Outbound connections

By default your instance can connect anywhere. You can change that (or make it so if it seems that the instance can't connect to the internet) to add a rule to the security group.

To do so, from the "Instances" tab, go to the far right of your running instance's line and click on its security group link. Then click on the bottom "Outbound" tab and edit the rules. If you see `All traffic | All | All | 0.0.0.0/0` it means, obviously, that the instance can do whatever it wants!


## 4. GUI Text editor

You can use Sublime Text **2** (not 3 saddly) to edit your remote files from your own computer using **[rsub](https://github.com/Drarok/rsub)** See this [tutorial](http://log.liminastudio.com/writing/tutorials/sublime-tunnel-of-love-how-to-edit-remote-files-with-sublime-text-via-an-ssh-tunnel). This means Sublime Text will edit the remote file using `scp` under the hood so you can use your GUI for the EC2 instance.

## 5. Testing

#### Tensorflow's examples
You can check that everyting runs by going to Tensorflow's examples : `cd ~/anaconda3/lib/python3.5/site-packages/tensorflow/models/image/cifar10` then either `python -m cifar10_train.py` or open an iPython console and `run cifar10_train.py`.
Tensorflow will first download the data it needs to train and then train displaying this kind of line :

> `2016-12-03 18:41:43.992273: step 100, loss = 4.08 (790.8 examples/sec; 0.162 sec/batch)`

You can check that this training speed is quite good compared to what they get [here](https://www.tensorflow.org/versions/r0.12/tutorials/deep_cnn/index.html).

#### Your own work
Once you're logged in your instance, you're basically within a GUI-free Ubuntu machine. Using the `scp` commands described above, you can transfer code and check that it runs as expected (or better!). Also using rsub is quite handy. Hacky testing : `nano my_file.py` then paste your code, `ctr+X` to quit and save (say `yes`) and `python my_file.py`. 

If the instance lacks specific libraries, well just like at home you can `apt-get` and `pip3` (etc.) what you need. However if you terminate the instance whithout saving a snapshot all personal settings will be gone when restarting later from my AMI. See next section. Suggest improvements if you feel like other people are going to need this library and it should be default.

## 6. End of work

You've done some nonesense for a while, now playtime is over. If you keep your instance running Amazon's going to keep billing you. You can either stop or terminate your instance. 

#### Terminate
You don't need this instance anymore, all the data it contains is going to be deleted. Consider taking a snapshot to backup your data (of course it's stored on Amazon, on S3, so it will be charged but cheaper than EBS). No more billing related to the instance however.

#### Stop
You'll re-use this instance soon enough. Its volumes are kept in Amazon Elastic Block Store (EBS). No more instance billing and your data's still here when you restart it (right click Instance state -> start) but you pay for EBS which is more expensive. But not that much if you have only a few GBs and not big TBs.

#### So...?

* Terminate if the job is done. Billing = zero.
* Termninate and snapshot if job is done but someday you'll need it again. Billing = compressed snapshot on S3.
* Stop if it is a recurrent work. billing = volume on EBS.

[More](http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-instance-lifecycle.html) and [more](http://docs.rightscale.com/faq/clouds/aws/Whats_the_difference_between_Terminating_and_Stopping_an_EC2_Instance.html) and [more](https://images.duckduckgo.com/iu/?u=https%3A%2F%2Fs-media-cache-ak0.pinimg.com%2F736x%2Ff2%2Fde%2F6d%2Ff2de6d3610642b866edcf76f7f86129a.jpg&f=1).

## 7. Improvements + Update form

You can either suggest improvements [here](https://forums.aws.amazon.com/thread.jspa?threadID=244014), build your own AMI from this one or from scratch -> see [here](http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AMIs.html). 

As far as this very tutorial is concerned, please pull-request edits, corrections, improvement suggestions and use issues to get help. I'm however far from being experienced with this... Let's hope there will be someone somewhere in the community  to help you. Or stackoverflow may be a good idea.

Also I will maintain a TO DO list for the next version of the AMI according to your feedback. I am not sure how this is going to evolve so if you want to be updated sign up to this **[form](https://goo.gl/forms/rutdrahXGJHLvO9h1)**. The use I'll have of this will be mainly to tell you when the next version is out but the most important thing is **I may delete the AMI** (I'm paying for it so I won't have too many of these) after such an upgrade. At which point you may want to snapshot your work to start on the new one so you'll need to know when this is going to happen. 

## 8. Pricing

I won't go into details here but roughly speaking you are charged **per hour** and according to the volume of your (S3) **snapshots** and (EBS) **volumes**. 

Checkout "My Billing Dashboard" when you click on your account in the top right corner.
