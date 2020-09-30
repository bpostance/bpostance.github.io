---
layout: page
title: Posts
permalink: /posts/
priority: 0.5
---

<ul>
  {% for post in site.posts %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a>
	  {{ post.excerpt }}
    </li>
  {% endfor %}
</ul>