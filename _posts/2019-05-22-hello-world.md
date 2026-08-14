---
layout: post
title:  "Hello Jekyll!"
date:   2019-05-23 20:56:21 +0000
categories: [blog-post,engineering]
tags: [general]
math: true
comments: true
---
Hello and welcome to my new Research and DataScience portfolio page. 

**Update March 2021**
I have now adopted the [Chirpy theme](https://chirpy.cotes.info/).

On this site I'll be posting links to projects, demos, and training material on machine learning and data science, in addition to research article reviews and other updates. I already have lots of content in my various [GitHub repos](https://github.com/bpostance), and this is my attempt to curate this and new work.

I have built this site using free hosting on [github.io](https://pages.github.com/) and the open-source site generator [Jekyll][jekyll-gh] written in Ruby.

I chose Jekyll because posts can be written in HTML or in Markdown (including Jupyter Notebooks), which is great if you create notebooks in R or Python.
To add new posts you simply add a file in the `_posts` directory that follows the convention `YYYY-MM-DD-name-of-post.markdown`. 
Take a look at the source for this post to get an idea about how it works.

Jekyll also offers powerful support for code snippets:

{% highlight ruby %}
def print_hi(name)
  puts "Hi, #{name}"
end
print_hi('Ben')
#=> prints 'Hi, Ben' to STDOUT.
{% endhighlight %}

[Latex math](https://stackoverflow.com/a/57370526/4538066):

$$ P(A|B) = \frac {P(B|A)P(A)}{P(B)} $$

Jekyll also supports:
 - [Google Analytics](https://desiredpersona.com/google-analytics-jekyll/)
 - [User comments with Just-comments](https://60devs.com/adding-comments-to-your-jekyll-blog.html) or [Disqus](https://desiredpersona.com/disqus-comments-jekyll/).
 - [Displaying Jupyter notebooks](https://www.linode.com/docs/applications/project-management/jupyter-notebook-on-jekyll/)

[jekyll-gh]:   https://github.com/jekyll/jekyll
