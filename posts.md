---
layout: page
title: Posts
permalink: /posts/
priority: 0.5
---


<!--
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
-->

<ul class="post-list">
  {%- for post in site.posts -%}
  <li>
    {%- assign date_format = site.minima.date_format | default: "%b %-d, %Y" -%}
    <span class="post-meta">{{ post.date | date: date_format }}</span>
    <h3>
      <a class="post-link" href="{{ post.url | relative_url }}">
        {{ post.title | escape }}
      </a>
    </h3>
    {%- if site.show_excerpts -%}
      {{ post.excerpt }}
    {%- endif -%}
  </li>
  {%- endfor -%}
</ul>