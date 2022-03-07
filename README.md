# ICog-Labs
I have officially joined icog labs internship program since January 2022.<br>

# Sefineh Tesfa<br>

# Feb 4 - 11,2022<br>

I have completed Introduction to Deep Learning with Pytorch, Udacity.<br>
I have been learning transfer learning using pytorch.<br>
I have practiced on image classification by using MNIST and FashionMNIST datasets.<br>
I have spent lots of time understanding the scene behind Neural Network, Just the detailed concept, not much more about the programming aspects.<br>
Web scraping by using BeautifulSoup and requests module in python.<br>
# Feb 11 - 26,2022
Just practicing i.e Transfer learning by using Pre Trained deep learning models like VGG16, resnet. <br>
Problem solving in LeetCode.com and HackerRank.com online problem solving challenge platforms.<br>
Watching youtube videos that are related to Deep Learning. <br>

# Feb 26 - 28, 2022<br>
Recommendation system using pytorch in python. <br>
Collaborative filtering and Content based filtering.<br>
User-based filtering and Item based filtering.<br>
I have been practicing recommendation system with Neural Network  by using MovieLens dataset and Pytorch low level API.<br>
I have been studying about Docker and WordPress.<br>

# March 2-3, 2022
Recommender System Using Amazon Reviews.<br>
Model-based collaborative filtering system.<br>

# Recommendation system 
<a href="https://www.jiristodulka.com/post/recsys_cf/">recommendation system</a>


# March 3-4, 2022
Recommendation system tutorial at LinkedIn.com the link is (https://www.linkedin.com/learning/machine-learning-and-ai-foundations-recommendations/recommend-by-predicting-missing-user-ratings?autoSkip=true&autoplay=true&resume=false)

 Sentimental analysis using Natural Language processing 
# March 4,2022
Starting Mathematics for Machine learning course on Coursera.
# March 4-6,2022
Completed Mathematics for Machine learning course week 1-week 4
# March 6, 2022 
Practicing recommender system using pytorch (at the office)
 




<h1 style="color:orange;">Types of the Systems</h1>
There are many ways and complex algorithms used to build a recommender system. The following are fundamental approaches. While reading, the reader should think which one may be the most effective method when it comes to a movie recommendation.

# The Most Popular Item: 
It is the simplest strategy and requires no coding skills. It works based on the assumption that the most popular item attracts most consumers or most users. For example, any consumer shopping on Amazon would see the most frequently bought items. Conversely, Netflix would recommend every user the most popular movie in its list.

Association & Market Based Model: The system makes recommendations based on the items in the consumer's basket. For instance, if the system detected that the buyer is purchasing ground coffee it would also suggest her to buy filters as well (observed association coffee - filters).

# Content Filtering: 
Uses metadata to determine the user's taste. For example, the system recommends the user movies based on their preferences of genres, actors, themes, etc. Such a system matches the user and the item based on similarity. For example, if the user watched and liked Terminator and Predator (both action movies with Arnold Schwarzenegger in the main role), it would probably recommend them to watch Commando.

# Collaborative Filtering (CF): 
It is an algorithmic architecture that recommends consumers items based on their observed behavior. There are two types of Collaborative Filtering frameworks: Model-Based Approach and Memory-Based Approach:

# User-based (UBCF): 
It is a predecessor of Item-based CF. UBCF makes recommendations based on the user's characteristics that are similar to other users in the system. For example, if the end-user positively rates a movie, the algorithm finds other users who have previously rated the movie too, i.e. these users are similar to one another. In the next step, the system recommends the user an unseen movie but highly rated by other - referenced - users. See Figure 1.
# Item-based (IBCF): 
IBCF was originally developed by Amazon and is currently adopted by most online corporations (e.g. Netflix, YouTube, etc.).

# Hybrid Models: 
As the name suggests, the Hybrid Models combine two or more recommendation strategies. For instance, a Hybrid Content-Collaborative System can recommend the user a movie based on their gender but still focuses on the movie features the user exhibits to prefer.
##### With the ever-growing volume of online information, recommender systems have been an effective strategy to overcome such information overload. The utility of recommender systems cannot be overstated, given its widespread adoption in many web applications, along with its potential impact to ameliorate many problems related to over-choice.
# Collaborative Filtering: Model-Based Approach
Once again, this article discusses Collaborative Item-based Filtering and focuses on the Model-Based Approach which tackles the two challenges imposed by CF. Unlike Memory-Based Approach, Model-Based procedure facilitates machine learning techniques such as Singular Value Decomposition (SVD) and Matrix Factorization models to predict the end user's rating on unrated items. In the context of a movie-to-movie recommender, a collaborative filter answers the question: “What movies have a similar user-rating profile?"(Lineberry & Longo, 2018).

# Matrix Factorization
Hopcroft and Kannan (2012), explains the whole concept of matrix factorization on customer data where m customers buy n products. The authors explain collaborative filtering in a comprehensive language. For demonstrative purposes, the author of this article demonstrates the concept on a specific case.

Let matrix 
R
m
∗
n
 represent the ratings on movies assigned by each user, also called the utility matrix. Specifically, the value 
r
i
j
=
5
 represents the rating of user i assigned to movie j. However, the individual's preference is determined by k factors. For example, the user's age, sex, income, education, etc. are likely to affect the user's behavior. Accordingly, the individual's rating of a movie (
r
i
j
) is determined by some weighted combinations of the hidden factors. In practice, customer's behavior can be characterized by a k-dimensional vector with much lower dimensions than the original matrix 
R
 with 
m
∗
n
 dimensions. The vector's components, also called the latent factors, represent the weight of each factor. For example, given a vector 
v
2
=
[
0.2
,
0.8
]
 it can be hypothesized that there are only two (unknown) latent factors with subsequent weights describing the rating (behavior).

Matrix factorization is an effective CF technique because it benefits from the properties of linear algebra. Specifically, consider matrix 
R
 as a record of various elements. As it is possible to decompose any integer into the product of its prime factor, matrix factorization also enables humans to explore information about matrices and their functional properties an array of elements (Goodfellow, Bengio, 2016)


# Singular Value Decomposition (SVD)
SVD decomposes any matrix into singular vectors and singular values. If the reader has previous experience with machine learning, particularly with dimensionality reduction, they would find traditional use of SVD in Principal Component Analysis (PCA). Simply put, SVD is equivalent to PCA after mean centering, i.e. shifting all data points so that their mean is on the origin (Gillis, 2014).

Formally, SVD is decomposition of a matrix R into the product of three matrices: 
Rm∗n=Um∗m Dm∗n <span>V<sup>t</sup></span> n∗n.
