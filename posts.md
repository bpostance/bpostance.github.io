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
      {%- assign date_format = site.minima.date_format | default: "%b %-d, %Y" -%}
      <p href="{{ post.url }}">{{ post.date | date: date_format }}</p>
	    {{ post.excerpt }}
    </li>
  {% endfor %}
</ul>